//===- OptimizeMILP.cpp - optimize MILP model over CFDFC  -------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <fstream>
#include <iostream>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

unsigned buffer::getPortInd(Operation *op, Value val) {
  for (auto [indVal, port] : llvm::enumerate(op->getResults())) {
    if (port == val) {
      return indVal;
    }
  }
  return UINT_MAX;
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

namespace {
/// Data structure to store the variables w.r.t to a unit(operation), including
/// whether it belongs to a CFDFC, and its retime variables.
struct UnitVar {
  bool select;
  GRBVar retIn, retOut;
};

/// Data structure to store the variables w.r.t to a channel(value), including
/// whether it belongs to a CFDFC, and its time, throughput, and buffer
/// placement decision.
struct ChannelVar {
  bool select;
  GRBVar tDataIn, tDataOut, tElasIn, tElasOut;
  GRBVar bufIsOp, bufNSlots, hasBuf;

  GRBVar tValidIn, tValidOut, tReadyIn, tReadyOut;
  GRBVar valbufIsOp, rdybufIsTr;
};

} // namespace

/// Get the pointer to the channel that defines the channel variables
static std::optional<Value *>
inChannelMap(const std::map<Value *, ChannelVar> &channelVars, Value ch) {
  for (auto &[chVal, chVar] : channelVars) {
    if (*chVal == ch)
      return chVal;
  }
  return nullptr;
}

/// Whether the path is considered to be covered in path and elasticity
/// constraints. Current version only consider mem_controller, future version
/// should take account of lsq and more operations.
static bool coverPath(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  Operation *dstOp = *(channel.getUsers().begin());
  // If both src operation and dst operation exist, and neither of them is
  // memory controller unit, the channel is covered.
  if (srcOp && dstOp)
    if (isa<MemoryControllerOp>(srcOp) || isa<MemoryControllerOp>(dstOp))
      return false;
  return true;
}

/// Initialize the variables in the MILP model
static void
initVarsInMilp(handshake::FuncOp funcOp, GRBModel &modelBuf,
               std::vector<CFDFC> cfdfcList, std::vector<unsigned> cfdfcInds,
               std::vector<Value> &allChannels,
               std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
               std::vector<DenseMap<Value, GRBVar>> &chThrptToks,
               DenseMap<Value, ChannelVar> &channelVars,
               std::map<std::string, UnitInfo> unitInfo, bool fpl22 = false) {
  // create variables
  for (auto [ind, cfdfc] : llvm::enumerate(cfdfcList)) {
    unitVars.emplace_back();
    chThrptToks.emplace_back();
    for (auto [unitInd, unit] : llvm::enumerate(cfdfc.units)) {
      UnitVar unitVar;

      // init unit variables
      std::string unitName = getOperationShortStrName(unit);
      unitVar.retIn =
          modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                          "mg" + std::to_string(ind) + "_inRetimeTok_" +
                              unitName + std::to_string(unitInd));
      if (getUnitLatency(unit, unitInfo) < 1e-10)
        unitVar.retOut = unitVar.retIn;
      else
        unitVar.retOut =
            modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                            "mg" + std::to_string(ind) + "_outRetimeTok_" +
                                unitName + std::to_string(unitInd));
      unitVars[ind][unit] = unitVar;
    }

    // init channel variables w.r.t the optimized CFDFC
    if (std::find(cfdfcInds.begin(), cfdfcInds.end(), ind) == cfdfcInds.end())
      continue;
    for (auto [chInd, channel] : llvm::enumerate(cfdfc.channels)) {
      std::string srcName = "arg_start";
      std::string dstName = "arg_end";
      Operation *srcOp = channel.getDefiningOp();
      Operation *dstOp = *channel.getUsers().begin();
      // Define the channel variable names w.r.t to its connected units
      if (srcOp)
        srcName = getOperationShortStrName(srcOp);
      if (dstOp)
        dstName = getOperationShortStrName(dstOp);

      std::string chName = "mg" + std::to_string(ind) + "_" + srcName + "_" +
                           dstName + "_" + std::to_string(chInd);
      chThrptToks[ind][channel] = modelBuf.addVar(
          0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "thrpt_" + chName);
    }
  }
  modelBuf.update();

  // create channel vars
  for (auto [ind, val] : llvm::enumerate(allChannels)) {
    Operation *srcOp = val.getDefiningOp();
    Operation *dstOp = *val.getUsers().begin();

    if (!srcOp && !dstOp)
      continue;

    std::string srcOpName = "arg_input";
    std::string dstOpName = "arg_end";

    // Define the channel variable names w.r.t to its connected
    // units
    if (srcOp)
      srcOpName = getOperationShortStrName(srcOp);
    if (dstOp)
      dstOpName = getOperationShortStrName(dstOp);

    // create channel variable
    ChannelVar channelVar;

    std::string chName =
        srcOpName + "_" + dstOpName + "_" + std::to_string(ind);

    channelVar.tDataIn = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                         "timePathIn_" + chName);
    channelVar.tDataOut = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                          "timePathOut_" + chName);

    channelVar.tElasIn = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                         "timeElasticIn_" + chName);
    channelVar.tElasOut = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                          "timeElasticOut_" + chName);

    channelVar.bufNSlots = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER,
                                           chName + "_bufNSlots");

    channelVar.hasBuf =
        modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, chName + "_hasBuf");
    channelVar.bufIsOp =
        modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, chName + "_bufIsOp");

    if (fpl22) {
      channelVar.tValidIn = modelBuf.addVar(
          0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "timeValidIn_" + chName);

      channelVar.tValidOut = modelBuf.addVar(
          0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "timeValidOut_" + chName);
      channelVar.tReadyIn = modelBuf.addVar(
          0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "timeReadyIn_" + chName);
      channelVar.tReadyOut = modelBuf.addVar(
          0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "timeReadyOut_" + chName);

      channelVar.valbufIsOp = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY,
                                              chName + "_valIsOpa");
      channelVar.rdybufIsTr = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY,
                                              chName + "_rdyIsTrans");
    }
    channelVars[val] = channelVar;
    modelBuf.update();
  }
}

/// Create time path constraints over channels.
/// t1 is the input time of the channel, t2 is the output time of the channel.
static void createPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                              GRBVar &bufOp, double period) {
  modelBuf.addConstr(t1 <= period);
  modelBuf.addConstr(t2 <= period);
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufOp);
}

// Data path constratins for a channel
// t1, t2 are the time stamps in the source and destination of the channel
// bufOp: whether the inserted buffer in a channel is opaque or not;
// bufRdyOp: whether the inserted buffer in a channel for the handshake ready
// signal is opaque or not;
// period: circuit period
// constrInd: index of the constraint
// bufDelay: delay of the inserted buffer in the channel
// ctrlBufDelay: delay of the inserted buffer for the handshake ready signal
static void createDataPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                  GRBVar &bufOp, GRBVar &bufRdyOp,
                                  double period, double bufDelay = 0.0,
                                  double ctrlBufDelay = 0.1) {
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufOp + ctrlBufDelay * bufRdyOp);
  // modelBuf.addConstr(t2 >= bufDelay);
  modelBuf.addConstr(t2 >= ctrlBufDelay * bufRdyOp);
}

static void createReadyPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                   GRBVar &bufOp, GRBVar &bufRdyOp,
                                   GRBVar &bufNSlots, double period,
                                   double bufDelay = 0.0,
                                   double ctrlBufDelay = 0.1) {
  modelBuf.addConstr(t1 <= period);
  modelBuf.addConstr(t2 <= period);
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufRdyOp + ctrlBufDelay * bufOp);
  // modelBuf.addConstr(t2 >= bufDelay);
  modelBuf.addConstr(t2 >= ctrlBufDelay * bufOp);
  modelBuf.addConstr(bufNSlots >= bufOp + bufRdyOp);
}

static void createValidPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                   GRBVar &bufValOp, GRBVar &bufRdyOp,
                                   GRBVar &bufOp, GRBVar &bufNSlots,
                                   double period, double bufDelay = 0.0,
                                   double ctrlBufDelay = 0.1) {
  modelBuf.addConstr(t1 <= period);
  modelBuf.addConstr(t2 <= period);
  modelBuf.addConstr(t2 >=
                     t1 - 2 * period * bufValOp + ctrlBufDelay * bufRdyOp);
  // modelBuf.addConstr(t2 >= bufDelay);
  modelBuf.addConstr(t2 >= ctrlBufDelay * bufRdyOp);
  //  buffer consistency constraints
  modelBuf.addConstr(bufValOp == bufOp);
}

// create control path constraints through a unit
static void createCtrlPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                  double delay) {
  modelBuf.addConstr(t2 >= t1 + delay);
}

// create elasticity constraints w.r.t channels
static void createElasticityConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                    GRBVar &bufOp, GRBVar &bufNSlots,
                                    GRBVar &hasBuf, unsigned cstCoef,
                                    double period) {
  modelBuf.addConstr(t2 >= t1 - cstCoef * bufOp);
  modelBuf.addConstr(bufNSlots >= bufOp);
  modelBuf.addConstr(hasBuf >= 0.01 * bufNSlots);
}

/// Throughput constraints over a channel
static void createThroughputConstrs(GRBModel &modelBuf, GRBVar &retSrc,
                                    GRBVar &retDst, GRBVar &thrptTok,
                                    GRBVar &thrpt, GRBVar &isOp,
                                    GRBVar &bufNSlots, const int tok) {
  modelBuf.addConstr(retSrc - retDst + thrptTok == tok);
  modelBuf.addConstr(thrpt + isOp - thrptTok <= 1);
  modelBuf.addConstr(thrptTok + thrpt + isOp - bufNSlots <= 1);
  modelBuf.addConstr(thrptTok - bufNSlots <= 0);
}

/// Create constraints that describe the circuits behavior
static void createModelPathConstraints(
    GRBModel &modelBuf, double targetCP, handshake::FuncOp &funcOp,
    std::vector<Value> &allChannels, DenseMap<Value, ChannelVar> &channelVars,
    std::map<std::string, UnitInfo> unitInfo) {
  // Channel constraints
  for (Value ch : allChannels) {
    // update the model to get the lower bound and upper bound of
    // the vars
    modelBuf.update();
    if (!channelVars.contains(ch) || !coverPath(ch))
      continue;

    auto chVars = channelVars[ch];

    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;
    GRBVar &bufOp = chVars.bufIsOp;

    createPathConstrs(modelBuf, t1, t2, bufOp, targetCP);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    double delayData = getCombinationalDelay(&op, unitInfo, "data");

    double latency = getUnitLatency(&op, unitInfo);
    if (latency == 0)
      // iterate all input port to all output port for a unit
      for (auto inChVal : op.getOperands()) {
        // Define variables w.r.t to input port
        if (!channelVars.contains(inChVal))
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        // GRBVar &tElasIn = channelVars[inChVal].tElasOut;
        for (auto outChVal : op.getResults()) {
          if (!channelVars.contains(outChVal))
            continue;

          // Define variables w.r.t to output port
          GRBVar &tOut = channelVars[outChVal].tDataIn;
          modelBuf.addConstr(tOut >= delayData + tIn);
        }
      }
    // if the unit is pipelined
    else {
      // Define constraints w.r.t to input port
      for (auto inChVal : op.getOperands()) {
        std::string out = "out";
        double inPortDelay = getPortDelay(inChVal, unitInfo, out);
        if (!channelVars.contains(inChVal))
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        modelBuf.addConstr(tIn <= targetCP - inPortDelay);
      }

      // Define constraints w.r.t to output port
      for (auto outChVal : op.getResults()) {
        std::string in = "in";
        double outPortDelay = getPortDelay(outChVal, unitInfo, in);

        if (!channelVars.contains(outChVal))
          continue;

        GRBVar &tOut = channelVars[outChVal].tDataIn;
        modelBuf.addConstr(tOut == outPortDelay);
      }
    }
  }
}

/// Create constraints that describe the circuits behavior
static void createModelPathConstraints_fpl22(
    GRBModel &modelBuf, double targetCP, FuncOp &funcOp, unsigned mgInd,
    std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
    DenseMap<Value, ChannelVar> &channelVars,
    std::map<std::string, UnitInfo> &unitInfo) {
  // Channel constraints
  for (auto [ch, chVars] : channelVars) {
    // update the model to get the lower bound and upper bound of the vars
    modelBuf.update();

    // place buffers if maxinum buffer slots is larger then 0 and the channel
    // is selected
    if (!chVars.select || chVars.bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
      continue;

    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createPathConstrs(modelBuf, t1, t2, bufOp, targetCP);
  }

  // Units constraints
  for (auto [op, _] : unitVars[mgInd]) {
    double delayData = getCombinationalDelay(op, unitInfo, "data");
    double latency = getUnitLatency(op, unitInfo);

    // iterate all input port to all output port for a unit
    if (latency == 0)
      for (auto inChVal : op->getOperands()) {
        // Define variables w.r.t to input port
        double inPortDelay = getPortDelay(inChVal, unitInfo, "out");
        if (!channelVars[inChVal].select)
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        for (auto outChVal : op->getResults()) {
          // Define variables w.r.t to output port
          double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

          if (!channelVars[outChVal].select)
            continue;

          GRBVar &tOut = channelVars[outChVal].tDataIn;
          modelBuf.addConstr(tOut >= delayData + tIn);
        }
      }
    else {
      for (auto inChVal : op->getOperands()) {
        double inPortDelay = getPortDelay(inChVal, unitInfo, "out");
        if (!channelVars[inChVal].select)
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        modelBuf.addConstr(tIn <= targetCP - inPortDelay);
      }
      for (auto outChVal : op->getResults()) {
        // Define variables w.r.t to output port
        double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

        if (!channelVars[outChVal].select)
          continue;

        GRBVar &tOut = channelVars[outChVal].tDataIn;
        modelBuf.addConstr(tOut == outPortDelay);
      }
    }
  }
}

static LogicalResult createModelCtrlConstraints_fpl22(
    GRBModel &modelBuf, double targetCP, FuncOp &funcOp, unsigned mgInd,
    std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
    DenseMap<Value, ChannelVar> &channelVars,
    std::map<std::string, UnitInfo> &unitInfo) {
  // Channel constraints
  for (auto mode : std::vector<std::string>{"valid", "ready"})
    for (auto [ch, chVars] : channelVars) {
      // update the model to get the lower bound and upper bound of the vars
      modelBuf.update();

      // place buffers if maxinum buffer slots is larger then 0 and the channel
      // is selected
      if (!chVars.select || chVars.bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
        continue;

      GRBVar &t1 = chVars.tDataIn;
      GRBVar &t2 = chVars.tDataOut;

      GRBVar &tValid1 = chVars.tValidIn;
      GRBVar &tValid2 = chVars.tValidOut;
      GRBVar &tReady1 = chVars.tReadyOut;
      GRBVar &tReady2 = chVars.tReadyIn;

      GRBVar &bufValOp = chVars.valbufIsOp;
      GRBVar &bufRdyTr = chVars.rdybufIsTr;

      GRBVar &bufOp = chVars.bufIsOp;
      GRBVar &bufNSlots = chVars.bufNSlots;
      if (mode == "valid") {
        createValidPathConstrs(modelBuf, tValid1, tValid2, bufValOp, bufRdyTr,
                               bufOp, bufNSlots, targetCP);
        createDataPathConstrs(modelBuf, t1, t2, bufOp, bufRdyTr, targetCP);
      }

      if (mode == "ready")
        createReadyPathConstrs(modelBuf, tReady1, tReady2, bufOp, bufRdyTr,
                               bufNSlots, targetCP);
    }

  // Units constraints
  for (auto mode : std::vector<std::string>{"valid", "ready"})
    for (auto [op, _] : unitVars[mgInd]) {
      // double delayData = getCombinationalDelay(op, unitInfo, "data");
      double delayValid = getCombinationalDelay(op, unitInfo, "valid");
      double delayReady = getCombinationalDelay(op, unitInfo, "ready");
      double latency = getUnitLatency(op, unitInfo);

      // iterate all input port to all output port for a unit
      if (latency == 0)
        for (auto inChVal : op->getOperands()) {
          // Define variables w.r.t to input port
          if (!channelVars[inChVal].select)
            continue;

          GRBVar &tValidIn = channelVars[inChVal].tValidOut;
          GRBVar &tReadyIn = channelVars[inChVal].tReadyIn;
          // GRBVar &tElasIn = channelVars[inChVal].tElasOut;
          for (auto outChVal : op->getResults()) {
            // Define variables w.r.t to output port

            if (!channelVars[outChVal].select)
              continue;

            GRBVar &tValidOut = channelVars[outChVal].tValidIn;
            GRBVar &tReadyOut = channelVars[outChVal].tReadyOut;
            if (mode == "valid")
              createCtrlPathConstrs(modelBuf, tValidIn, tValidOut, delayValid);

            if (mode == "ready")
              createCtrlPathConstrs(modelBuf, tReadyIn, tReadyOut, delayReady);
          }
        }
    }
  return success();
}

static void
createMixedConstrs(GRBModel &modelBuf, unsigned mgInd,
                   std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
                   DenseMap<Value, ChannelVar> &channelVars,
                   std::map<std::string, UnitInfo> unitInfo) {

  // create mixed domain constraints
  for (auto [op, _] : (unitVars[mgInd])) {
    // if the unit is an operator
    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp,
            arith::CmpIOp, arith::DivSIOp, arith::ExtSIOp, arith::TruncIOp,
            BranchOp, MuxOp, ConditionalBranchOp>(op)) {
      if (double delayVR = getMixedDelay(op, unitInfo, "VR");
          delayVR > 0 && delayVR < 100) {
        for (auto channel1 : op->getOperands()) {
          for (auto channel2 : op->getOperands()) {
            if (isa<MuxOp>(op)) {
              if (channel1 == op->getOperand(0) ||
                  channel2 == op->getOperand(0))
                continue;
            } else if (channel1 == channel2)
              continue; // skip the same port

            if (channel1 && channel2) {
              GRBVar &tValidIn1 = channelVars[channel1].tValidOut;
              GRBVar &tReadyOut2 = channelVars[channel2].tReadyIn;
              llvm::errs() << "op: "
                           << unitVars[0][op].retIn.get(GRB_StringAttr_VarName)
                           << ": VR" << delayVR << "\n";
              createCtrlPathConstrs(modelBuf, tValidIn1, tReadyOut2, delayVR);
            }
          }
        }
      }
    }
    // if the unit is conditional branch or mux
    if (isa<ConditionalBranchOp, MuxOp>(op)) {
      unsigned opInd = 0;
      if (isa<ConditionalBranchOp>(op))
        opInd = 1;
      GRBVar &tIn = channelVars[op->getOperand(opInd)].tDataOut;
      for (auto outCh : op->getOperands()) {

        GRBVar &tValidOut = channelVars[outCh].tValidIn;
        if (double delayCV = getMixedDelay(op, unitInfo, "DV");
            delayCV > 0 && delayCV < 100) {
          // if (isa<ConditionalBranchOp>(op))
          //   if (outCh == op->getResult(1))
          //     continue; // skip false reasult
          llvm::errs() << "op: "
                       << unitVars[0][op].retIn.get(GRB_StringAttr_VarName)
                       << ": DV" << delayCV << "\n";
          createCtrlPathConstrs(modelBuf, tIn, tValidOut, delayCV);
        }
      }

      for (unsigned ind = opInd; ind < op->getNumOperands(); ind++)
        if (double delayCR = getMixedDelay(op, unitInfo, "DR");
            delayCR > 0 && delayCR < 100) {
          GRBVar &tReadyOut = channelVars[op->getOperand(ind)].tReadyIn;
          createCtrlPathConstrs(modelBuf, tIn, tReadyOut, delayCR);
          llvm::errs() << "op: "
                       << unitVars[0][op].retIn.get(GRB_StringAttr_VarName)
                       << ": DR" << delayCR << "\n";
        }
    }
    // if the unit is a control merge
    if (isa<ControlMergeOp>(op)) {
      auto channel1 = op->getOperand(0);
      auto channel2 = op->getResult(1);
      if (channel1 && channel2) {
        GRBVar &tValidIn = channelVars[channel1].tValidOut;
        GRBVar &tOut = channelVars[channel2].tDataIn;
        if (double delayVC = getMixedDelay(op, unitInfo, "VD");
            delayVC > 0 && delayVC < 100) {
          createCtrlPathConstrs(modelBuf, tValidIn, tOut, delayVC);
          llvm::errs() << "op: " << op->getName() << ": VD\n";
        }
      }
    }
  }
  // mixed domain
}

/// Create constraints that describe the circuits behavior
static void
createModelElasticityConstraints(GRBModel &modelBuf, double targetCP,
                                 FuncOp &funcOp, unsigned unitNum,
                                 DenseMap<Value, ChannelVar> &channelVars,
                                 std::map<std::string, UnitInfo> unitInfo) {
  // Channel constraints
  for (auto [ch, chVars] : channelVars) {
    // update the model to get the lower bound and upper bound of the vars
    modelBuf.update();

    // place buffers if maxinum buffer slots is larger then 0 and the channel
    // is selected
    if (chVars.bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
      continue;

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createElasticityConstrs(modelBuf, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            unitNum + 2, targetCP);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    // iterate all input port to all output port for a unit
    for (auto inChVal : op.getOperands()) {
      if (!channelVars.contains(inChVal))
        continue;

      // Define variables w.r.t to input port
      GRBVar &tElasIn = channelVars[inChVal].tElasOut;
      for (auto outChVal : op.getResults()) {
        if (!channelVars.contains(outChVal))
          continue;

        // Define variables w.r.t to output port
        GRBVar &tElasOut = channelVars[outChVal].tElasIn;
        modelBuf.addConstr(tElasOut >= 1 + tElasIn);
      }
    }
  }
}

/// Create constraints that describe the circuits behavior
static LogicalResult createModelElasticityConstraints_fpl22(
    GRBModel &modelBuf, double targetCP, FuncOp &funcOp, unsigned unitNum,
    std::vector<DenseMap<Operation *, UnitVar>> &unitVars, unsigned mgInd,
    DenseMap<Value, ChannelVar> &channelVars,
    std::map<std::string, UnitInfo> unitInfo) {
  // Channel constraints
  for (auto [ch, chVars] : channelVars) {
    // update the model to get the lower bound and upper bound of the vars
    modelBuf.update();

    // place buffers if maxinum buffer slots is larger then 0 and the channel
    // is selected
    if (!chVars.select || chVars.bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
      continue;

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createElasticityConstrs(modelBuf, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            unitNum + 1, targetCP);
  }

  // Units constraints
  for (auto [op, _] : unitVars[mgInd]) {
    // iterate all input port to all output port for a unit
    for (auto inChVal : op->getOperands()) {
      // Define variables w.r.t to input port
      if (!channelVars[inChVal].select)
        continue;

      GRBVar &tElasIn = channelVars[inChVal].tElasOut;
      for (auto outChVal : op->getResults()) {
        // Define variables w.r.t to output port
        if (!channelVars[outChVal].select)
          continue;

        GRBVar &tElasOut = channelVars[outChVal].tElasIn;
        modelBuf.addConstr(tElasOut >= 1 + tElasIn);
      }
    }
  }
  return success();
}

static void createModelThrptConstraints(
    GRBModel &modelBuf, std::vector<GRBVar> &circtThrpt,
    std::vector<DenseMap<Value, GRBVar>> &chThrptToks,
    std::vector<CFDFC> &cfdfcList, DenseMap<Value, ChannelVar> &channelVars,
    std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
    std::map<std::string, UnitInfo> &unitInfo) {
  for (auto [ind, subMG] : llvm::enumerate(chThrptToks)) {
    for (auto ch : cfdfcList[ind].channels) {

      if (!subMG.contains(ch))
        continue;

      Operation *srcOp = ch.getDefiningOp();
      Operation *dstOp = *(ch.getUsers().begin());

      GRBVar &thrptTok = subMG[ch];
      int tok = isBackEdge(srcOp, dstOp) ? 1 : 0;
      GRBVar &retSrc = unitVars[ind][srcOp].retOut;
      GRBVar &retDst = unitVars[ind][dstOp].retIn;

      GRBVar &bufOp = channelVars[ch].bufIsOp;
      GRBVar &bufNSlots = channelVars[ch].bufNSlots;
      createThroughputConstrs(modelBuf, retSrc, retDst, thrptTok,
                              circtThrpt[ind], bufOp, bufNSlots, tok);
    }
  }
  for (auto [ind, subMG] : llvm::enumerate(unitVars))
    for (auto &[op, unitVar] : subMG) {
      GRBVar &retIn = unitVar.retIn;
      GRBVar &retOut = unitVar.retOut;
      double latency = getUnitLatency(op, unitInfo);
      if (latency > 0)
        modelBuf.addConstr(retOut - retIn == latency * circtThrpt[ind]);
    }
}

/// Create constraints that is prerequisite for buffer placement
static void
setCustomizedConstraints(GRBModel &modelBuf,
                         DenseMap<Value, ChannelVar> &channelVars,
                         DenseMap<Value, ChannelBufProps> &channelBufProps,
                         DenseMap<Value, Result> &res) {
  for (auto &[ch, chVars] : channelVars) {
    // set min value of the buffer
    if (channelBufProps[ch].minOpaque > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minOpaque);
      modelBuf.addConstr(chVars.bufIsOp >= 0);
    } else if (channelBufProps[ch].minTrans > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minTrans);
      modelBuf.addConstr(chVars.rdybufIsTr >= 0);
      modelBuf.addConstr(chVars.bufIsOp <= 0);
    }

    // set max value of the buffer
    if (channelBufProps[ch].maxOpaque.has_value())
      modelBuf.addConstr(chVars.bufNSlots <=
                         channelBufProps[ch].maxOpaque.value());

    if (channelBufProps[ch].maxTrans.has_value())
      modelBuf.addConstr(chVars.bufNSlots <=
                         channelBufProps[ch].maxTrans.value());
  }
  for (auto &[ch, result] : res) {
    modelBuf.addConstr(channelVars[ch].bufNSlots >= res[ch].numSlots);
    modelBuf.addConstr(channelVars[ch].bufIsOp >= res[ch].opaque);
  }
}

// Create MILP cost function
static void createModelObjective(GRBModel &modelBuf,
                                 std::vector<GRBVar> &circtThrpts,
                                 std::vector<CFDFC> &cfdfcList,
                                 DenseMap<Value, ChannelVar> &channelVars) {
  GRBLinExpr objExpr;
  double lumbdaCoef1 = 1e-4;
  double lumbdaCoef2 = 1e-5;

  double totalFreq = 0.0;
  double highestCoef = 0.0;
  for (auto [channel, _] : channelVars)
    totalFreq += static_cast<double>(getChannelFreq(channel, cfdfcList));

  for (auto [ind, thrpt] : llvm::enumerate(circtThrpts)) {
    double coef =
        cfdfcList[ind].channels.size() * cfdfcList[ind].execN / totalFreq;
    highestCoef = std::max(coef, highestCoef);
    objExpr += coef * thrpt;
  }

  for (auto &[_, chVar] : channelVars) {
    objExpr -= highestCoef *
               (lumbdaCoef1 * chVar.hasBuf + lumbdaCoef2 * chVar.bufNSlots);
  }

  modelBuf.setObjective(objExpr, GRB_MAXIMIZE);
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

LogicalResult buffer::placeBufferInCFDFCircuit(
    DenseMap<Value, Result> &res, handshake::FuncOp &funcOp,
    std::vector<Value> &allChannels, std::vector<CFDFC> &cfdfcList,
    std::vector<unsigned> &cfdfcInds, double targetCP, int timeLimit,
    bool setCustom, std::map<std::string, UnitInfo> &unitInfo,
    DenseMap<Value, ChannelBufProps> &channelBufProps) {

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  llvm::errs() << "Project was built without Gurobi installed, can't run "
                  "buffer placement\n";
  return failure();
#else
  // create a Gurobi environment

  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel modelBuf = GRBModel(env);

  // create variables
  std::vector<DenseMap<Operation *, UnitVar>> unitVars;
  std::vector<DenseMap<Value, GRBVar>> chThrptToks;
  DenseMap<Value, ChannelVar> channelVars;
  std::vector<GRBVar> circtThrpts;

  // create the variable to noate the overall circuit throughput
  for (auto [ind, _] : llvm::enumerate(cfdfcList)) {
    GRBVar circtThrpt = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                        "thrpt" + std::to_string(ind));
    circtThrpts.push_back(circtThrpt);
  }

  // Compute the total numbers of units in the circuit as a coef for
  // elasticity constraints
  unsigned unitNum =
      std::distance(funcOp.getOps().begin(), funcOp.getOps().end());

  // initialize variables
  initVarsInMilp(funcOp, modelBuf, cfdfcList, cfdfcInds, allChannels, unitVars,
                 chThrptToks, channelVars, unitInfo);

  // define customized constraints
  if (setCustom)
    setCustomizedConstraints(modelBuf, channelVars, channelBufProps, res);

  // create circuits constraints
  createModelPathConstraints(modelBuf, targetCP, funcOp, allChannels,
                             channelVars, unitInfo);

  createModelElasticityConstraints(modelBuf, targetCP, funcOp, unitNum + 2,
                                   channelVars, unitInfo);

  createModelThrptConstraints(modelBuf, circtThrpts, chThrptToks, cfdfcList,
                              channelVars, unitVars, unitInfo);

  // create cost function
  createModelObjective(modelBuf, circtThrpts, cfdfcList, channelVars);

  modelBuf.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit);
  modelBuf.optimize();

  //  The result is optimal if the model is solved to optimality or the time
  //  limit and obtain position throughput
  bool isOptimal = (modelBuf.get(GRB_IntAttr_Status) == GRB_OPTIMAL) ||
                   (modelBuf.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT &&
                    modelBuf.get(GRB_DoubleAttr_ObjVal) > 0);

  if (!isOptimal) {
    llvm::errs() << "no optimal sol\n";
    return failure();
  }

  // load answer to the result
  for (auto &[ch, chVarMap] : channelVars) {
    if (chVarMap.hasBuf.get(GRB_DoubleAttr_X) > 0) {
      Result result;
      result.numSlots =
          static_cast<int>(chVarMap.bufNSlots.get(GRB_DoubleAttr_X) + 0.5);
      result.opaque = chVarMap.bufIsOp.get(GRB_DoubleAttr_X) > 0;
      res[ch] = result;
    }
  }
  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}

LogicalResult buffer::placeBufferInCFDFCircuit(
    DenseMap<Value, Result> &res, handshake::FuncOp funcOp,
    std::vector<Value> &allChannels, std::vector<CFDFC> cfdfcList,
    unsigned cfdfcInd, double targetCP, int timeLimit,
    std::map<std::string, UnitInfo> unitInfo,
    DenseMap<Value, ChannelBufProps> channelBufProps) {

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  llvm::errs() << "Project was built without Gurobi installed, can't run "
                  "buffer placement\n";
  return failure();
#else
  // create a Gurobi environment

  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel modelBuf = GRBModel(env);

  // create variables
  std::vector<DenseMap<Operation *, UnitVar>> unitVars;
  std::vector<DenseMap<Value, GRBVar>> chThrptToks;
  DenseMap<Value, ChannelVar> channelVars;
  std::vector<GRBVar> circtThrpts;

  // create the variable to noate the overall circuit throughput
  for (auto [ind, _] : llvm::enumerate(cfdfcList)) {
    GRBVar circtThrpt = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                        "thrpt" + std::to_string(ind));
    circtThrpts.push_back(circtThrpt);
  }

  // Compute the total numbers of units in the circuit as a coef for
  // elasticity constraints
  unsigned unitNum =
      std::distance(funcOp.getOps().begin(), funcOp.getOps().end());

  createModelPathConstraints_fpl22(modelBuf, targetCP, funcOp, cfdfcInd,
                                   unitVars, channelVars, unitInfo);

  createModelCtrlConstraints_fpl22(modelBuf, targetCP, funcOp, cfdfcInd,
                                   unitVars, channelVars, unitInfo);

  // createMixedConstrs(modelBuf, cfdfcInd, unitVars, channelVars,
  // unitInfo);

  createModelElasticityConstraints_fpl22(modelBuf, targetCP, funcOp, unitNum,
                                         unitVars, cfdfcInd, channelVars,
                                         unitInfo);

  createModelThrptConstraints(modelBuf, circtThrpts, chThrptToks, cfdfcList,
                              channelVars, unitVars, unitInfo);

  // create cost function
  createModelObjective(modelBuf, circtThrpts, cfdfcList, channelVars);

  modelBuf.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit);
  modelBuf.optimize();

  //  The result is optimal if the model is solved to optimality or the time
  //  limit and obtain position throughput
  bool isOptimal = (modelBuf.get(GRB_IntAttr_Status) == GRB_OPTIMAL) ||
                   (modelBuf.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT &&
                    modelBuf.get(GRB_DoubleAttr_ObjVal) > 0);

  if (!isOptimal) {
    llvm::errs() << "no optimal sol\n";
    return failure();
  }

  // load answer to the result
  for (auto &[ch, chVarMap] : channelVars) {
    if (chVarMap.hasBuf.get(GRB_DoubleAttr_X) > 0) {
      Result result;
      result.numSlots =
          static_cast<int>(chVarMap.bufNSlots.get(GRB_DoubleAttr_X) + 0.5);
      result.opaque = chVarMap.bufIsOp.get(GRB_DoubleAttr_X) > 0;
      res[ch] = result;
    }
  }
  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}
