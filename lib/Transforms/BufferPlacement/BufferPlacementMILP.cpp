//===- BufferPlacementMILP.cpp - MILP-based buffer placement ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements MILP-based buffer placement (requires Gurobi).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include <fstream>

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

namespace {

using NameToOp = std::unordered_map<std::string, Operation *>;
using OpToName = std::unordered_map<Operation *, std::string>;

struct ChannelInfo {
  std::string src;
  unsigned srcPort;
  std::string dst;
  unsigned dstPort;
  double delay;
  unsigned minTrans;
  std::optional<unsigned> maxTrans;
  unsigned minOpaque;
  std::optional<unsigned> maxOpaque;
};
} // namespace

// Ugly but who cares.
NameToOp names;
OpToName ops;

/// Determines whether a string is a valid unsigned integer.
static bool isUnsigned(const std::string &str) {
  return std::all_of(str.begin(), str.end(), [](char c) { return isdigit(c); });
}

static bool isDouble(const std::string &str) {
  return std::all_of(str.begin(), str.end(),
                     [](char c) { return isdigit(c) || c == '.'; });
}

static std::string strip(const std::string &str) {
  unsigned startIdx = 0, endIdx = str.size();
  bool leading = true;
  for (auto [idx, c] : llvm::enumerate(str)) {
    if (leading) {
      if (!std::isspace(c)) {
        leading = false;
        startIdx = idx;
      }
    } else {
      if (std::isspace(c)) {
        endIdx = idx;
        break;
      }
    }
  }
  return leading ? "" : str.substr(startIdx, endIdx - startIdx);
}

/// Kinda DFS path finding.
static bool findPath(Operation *src, Operation *dst,
                     SmallVector<mlir::Value> &path) {
  if (src == dst)
    return true;
  for (OpResult res : src->getResults()) {
    Operation *user = *res.getUsers().begin();
    // Have we found the destination
    if (user == dst) {
      path.push_back(res);
      return true;
    }
    // Detect loops
    if (llvm::any_of(path, [&](mlir::Value val) {
          return *val.getUsers().begin() == user;
        }))
      return false;
    SmallVector<mlir::Value> newPath;
    llvm::copy(path, std::back_inserter(newPath));
    if (findPath(user, dst, newPath)) {
      path = newPath;
      return true;
    }
  }
  return false;
}

static bool findPath(mlir::Value src, mlir::Value dst,
                     SmallVector<mlir::Value> &path) {
  path.push_back(src);
  bool ret = findPath(*src.getUsers().begin(), dst.getDefiningOp(), path);
  path.push_back(dst);
  return ret;
}

/// Transforms the port number associated to an edge endpoint to match the
/// operand ordering of legacy Dynamatic.
static size_t fixPortNumber(Operation *op, size_t idx, bool isSrcOp) {
  return llvm::TypeSwitch<Operation *, size_t>(op)
      .Case<handshake::ConditionalBranchOp>([&](auto) {
        if (isSrcOp)
          return idx;
        // Legacy Dynamatic has the data operand before the condition operand
        return 1 - idx;
      })
      .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
        // Legacy Dynamatic has the memory controls before the return values
        auto numReturnValues = endOp.getReturnValues().size();
        auto numMemoryControls = endOp.getMemoryControls().size();
        return (idx < numMemoryControls) ? idx + numReturnValues
                                         : idx - numMemoryControls;
      })
      .Case<handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>([&](auto) {
        // Legacy Dynamatic has the data operand/result before the address
        // operand/result
        return 1 - idx;
      })
      .Default([&](auto) { return idx; });
}

static BlockArgument getFunArg(handshake::FuncOp funcOp, StringRef srcName) {
  unsigned numArgs = funcOp.getNumArguments();
  for (size_t idx = 0; idx < numArgs; ++idx) {
    std::string argName =
        idx == numArgs - 1 ? "start_0" : funcOp.getArgName(idx).str();
    if (argName == srcName)
      return funcOp.getArguments()[idx];
  }
  return nullptr;
}

static bool channelCanExist(handshake::FuncOp funcOp, ChannelInfo &info) {
  // Identify source port
  if (!getFunArg(funcOp, info.src)) {
    if (names.find(info.src) == names.end()) {
      llvm::errs() << "No operation named " << info.src << " could be found\n";
      return false;
    }
    Operation *srcOp = names[info.src];
    size_t srcPortIdx = fixPortNumber(srcOp, info.srcPort - 1, true);
    if (srcPortIdx > srcOp->getNumResults()) {
      llvm::errs() << "Operation " << info.src << " has "
                   << srcOp->getNumResults() << " results but source port is "
                   << srcPortIdx << "";
      return false;
    }
  }

  // Identify destination port
  if (names.find(info.dst) == names.end()) {
    llvm::errs() << "No operation named " << info.dst << " could be found\n";
    return false;
  }
  Operation *dstOp = names[info.dst];
  size_t dstPortIdx = fixPortNumber(dstOp, info.dstPort - 1, false);
  if (dstPortIdx > dstOp->getNumOperands()) {
    llvm::errs() << "Operation " << info.src << " has "
                 << dstOp->getNumOperands()
                 << " operands but destination port is " << dstPortIdx << "";
    return false;
  }
  return true;
}

static bool findChannel(handshake::FuncOp funcOp, ChannelInfo &info,
                        mlir::Value &channel) {
  if (BlockArgument blockArg = getFunArg(funcOp, info.src); blockArg) {
    channel = blockArg;
  } else {
    Operation *srcOp = names[info.src];
    size_t srcPortIdx = fixPortNumber(srcOp, info.srcPort - 1, true);
    channel = srcOp->getResult(srcPortIdx);
  }

  Operation *dstOp = names[info.dst];
  size_t dstPortIdx = fixPortNumber(dstOp, info.dstPort - 1, false);
  return channel == dstOp->getOperand(dstPortIdx);
}

static LogicalResult annotateFunc(handshake::FuncOp funcOp,
                                  const std::string &filepath,
                                  PathConstraints &pathConstraints) {
  // Re-derive operation names in the same way as the DOT does
  std::map<std::string, unsigned> opTypeCntrs;
  for (auto &op : funcOp.getOps()) {
    std::string opFullName = op.getName().getStringRef().str();
    auto startIdx = opFullName.find('.');
    if (startIdx == std::string::npos)
      startIdx = 0;
    unsigned id;
    if (auto memOp = dyn_cast<handshake::MemoryControllerOp>(op))
      // Memories already have unique IDs, so make their name match it
      id = memOp.getId();
    else
      id = opTypeCntrs[op.getName().getStringRef().str()]++;
    std::string fullName = opFullName.substr(startIdx + 1) + std::to_string(id);
    names[fullName] = &op;
    ops[&op] = fullName;
  }

  // Open the buffer information file
  std::ifstream inputFile(filepath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open buffer information\n";
    return failure();
  }

  std::string token;

  auto parseDouble = [&](std::istringstream &iss,
                         double &value) -> ParseResult {
    std::getline(iss, token, ',');
    token = strip(token);
    if (!isDouble(token))
      return failure();
    value = std::stod(token);
    return success();
  };

  auto parseUnsigned = [&](std::istringstream &iss,
                           unsigned &value) -> ParseResult {
    std::getline(iss, token, ',');
    token = strip(token);
    if (token.empty()) {
      value = 0;
      return success();
    }
    if (!isUnsigned(token))
      return failure();
    value = std::stoi(token);
    return success();
  };

  auto parseOptUnsigned = [&](std::istringstream &iss,
                              std::optional<unsigned> &value) -> ParseResult {
    std::getline(iss, token, ',');
    token = strip(token);
    if (token.empty()) {
      value = std::nullopt;
      return success();
    }
    if (!isUnsigned(token))
      return failure();
    value = std::stoi(token);
    return success();
  };

  auto parseString = [&](std::istringstream &iss,
                         std::string &value) -> ParseResult {
    std::getline(iss, token, ',');
    value = strip(token);
    return success();
  };

  auto mergeProps = [&](Channel &channel,
                        ChannelBufProps &newProps) -> LogicalResult {
    ChannelBufProps &oldProps = *channel.props;

    std::optional<unsigned> maxTrans;
    if (!oldProps.maxTrans.has_value())
      maxTrans = newProps.maxTrans;
    else if (!newProps.maxTrans.has_value())
      maxTrans = oldProps.maxTrans;
    else
      maxTrans = std::min(*oldProps.maxTrans, *newProps.maxTrans);

    std::optional<unsigned> maxOpaque;
    if (!oldProps.maxOpaque.has_value())
      maxOpaque = newProps.maxOpaque;
    else if (!newProps.maxOpaque.has_value())
      maxOpaque = oldProps.maxOpaque;
    else
      maxOpaque = std::min(*oldProps.maxOpaque, *newProps.maxOpaque);

    unsigned minTrans = std::max(oldProps.minTrans, newProps.minTrans);
    if (maxTrans.has_value() && minTrans > *maxTrans)
      minTrans = *maxTrans;

    unsigned minOpaque = std::max(oldProps.minOpaque, newProps.minOpaque);
    if (maxOpaque.has_value() && minOpaque > *maxOpaque)
      minOpaque = *maxOpaque;

    ChannelBufProps combined(minTrans, maxTrans, minOpaque, maxOpaque);
    return dynamatic::buffer::replaceBufProps(channel.value, combined);
  };

  // Skip the header line
  std::string line;
  std::getline(inputFile, line);

  // Parse lines one by one, creating an ArchBB for each
  size_t idx = 1;
  while (std::getline(inputFile, line)) {
    idx++;
    std::istringstream iss(line);

    // Parse all columns
    ChannelInfo info;
    if (parseString(iss, info.src) || parseUnsigned(iss, info.srcPort) ||
        parseString(iss, info.dst) || parseUnsigned(iss, info.dstPort) ||
        parseDouble(iss, info.delay) || parseUnsigned(iss, info.minTrans) ||
        parseOptUnsigned(iss, info.maxTrans) ||
        parseUnsigned(iss, info.minOpaque) ||
        parseOptUnsigned(iss, info.maxOpaque)) {
      llvm::errs() << "Failed to parse constraint on line " << idx << "\n"
                   << line << "\n";
      return failure();
    }

    ChannelBufProps props(info.minTrans, info.maxTrans, info.minOpaque,
                          info.maxOpaque, info.delay);

    // If no source and destination components are provided, interpret the
    // constraints as global
    if (info.src.empty() && info.dst.empty()) {
      for (BlockArgument arg : funcOp.getArguments())
        for (Operation *user : arg.getUsers()) {
          Channel channel(arg, *funcOp, *user, true);
          if (failed(mergeProps(channel, props))) {
            llvm::errs() << "Failed to impose global channel constraint.\n";
            return failure();
          }
        }

      for (Operation &op : funcOp.getOps())
        for (OpResult res : op.getResults())
          for (Operation *user : res.getUsers()) {
            Channel channel(res, op, *user, true);
            if (failed(mergeProps(channel, props))) {
              llvm::errs() << "Failed to impose global channel constraint.\n";
              return failure();
            }
          }
      continue;
    }

    if (!channelCanExist(funcOp, info))
      return failure();

    // Try to find the value that is refered to by the line
    mlir::Value channel;
    if (findChannel(funcOp, info, channel)) {
      llvm::errs() << "[INFO] Identified constraint on line " << idx
                   << " as directly connected channel"
                   << "\n";

      // Set buffering properties for the channel
      if (failed(dynamatic::buffer::replaceBufProps(channel, props)))
        return failure();
    } else {
      // Failing that, try to find a path between source and destination
      SmallVector<mlir::Value> path;
      Value resVal;
      if (BlockArgument blockArg = getFunArg(funcOp, info.src); blockArg) {
        resVal = blockArg;
      } else {
        Operation *srcOp = names[info.src];
        size_t srcPortIdx = fixPortNumber(srcOp, info.srcPort - 1, true);
        resVal = srcOp->getResult(srcPortIdx);
      }
      Operation *dstOp = names[info.dst];
      size_t dstPortIdx = fixPortNumber(dstOp, info.dstPort - 1, false);
      if (!findPath(resVal, dstOp->getOperand(dstPortIdx), path)) {
        llvm::errs() << "Failed to find channel or path corresponding to "
                        "constraint on line "
                     << idx << "\n";
        return failure();
      }
      pathConstraints.push_back(std::make_pair(path, props));
      llvm::errs() << "[INFO] Identified constraint on line " << idx
                   << " as path: " << ops[path[0].getDefiningOp()];
      for (mlir::Value val : path)
        llvm::errs() << " -> " << ops[*val.getUsers().begin()];
      llvm::errs() << "\n";
    }
  }
  return success();
}

BufferPlacementMILP::BufferPlacementMILP(FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         const std::string &constraints,
                                         double targetPeriod, double maxPeriod,
                                         GRBEnv &env, Logger *logger)
    : timingDB(timingDB), targetPeriod(targetPeriod), maxPeriod(maxPeriod),
      funcInfo(funcInfo), model(GRBModel(env)), logger(logger) {

  if (!constraints.empty()) {
    if (failed(annotateFunc(funcInfo.funcOp, constraints, pathConstraints))) {
      llvm::errs() << "Failed to add extra constraints.\n";
      unsatisfiable = true;
      return;
    }
  }

  // Give a unique name to each operation
  std::map<std::string, unsigned> instanceNameCntr;
  for (Operation &op : funcInfo.funcOp.getOps()) {
    std::string shortName = op.getName().stripDialect().str();
    nameUniquer[&op] =
        shortName + std::to_string(instanceNameCntr[shortName]++);
  }

  // Combines any channel-specific buffering properties coming from IR
  // annotations to internal buffer specifications and stores the combined
  // properties into the channel map. Fails and marks the MILP unsatisfiable
  // if any of those combined buffering properties become unsatisfiable.
  auto deriveBufferingProperties = [&](Channel &channel) -> LogicalResult {
    // Increase the minimum number of slots if internal buffers are present,
    // and check for satisfiability
    if (failed(addInternalBuffers(channel))) {
      unsatisfiable = true;
      std::stringstream stream;
      stream << "For channel " << getChannelName(channel.value)
             << ": including internal component buffers into buffering "
                "properties of outgoing channel made them unsatisfiable. "
                "Properties are "
             << *channel.props;
      return channel.producer.emitError() << stream.str();
    }
    channels[channel.value] = *channel.props;
    return success();
  };

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcInfo.funcOp.getArguments())) {
    Channel channel(arg, *funcInfo.funcOp, **arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return;
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcInfo.funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      Channel channel(res, op, **res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel)))
        return;
    }
  }
}

bool BufferPlacementMILP::arePlacementConstraintsSatisfiable() {
  return !unsatisfiable;
}

LogicalResult BufferPlacementMILP::setup() {
  if (failed(createVars()))
    return failure();

  std::vector<Value> allChannels, nonMemChannels;
  for (auto &[channel, _] : channels) {
    allChannels.push_back(channel);
    if (!isa_and_nonnull<handshake::MemoryControllerOp>(
            channel.getDefiningOp()) &&
        !isa<handshake::MemoryControllerOp>(*channel.getUsers().begin()))
      nonMemChannels.push_back(channel);
  }
  std::vector<Operation *> allUnits;
  for (Operation &op : funcInfo.funcOp.getOps())
    allUnits.push_back(&op);

  if (failed(addCustomChannelConstraints(allChannels)) ||
      failed(addPathConstraints(nonMemChannels, allUnits)) ||
      failed(addElasticityConstraints(nonMemChannels, allUnits)))
    return failure();

  // Add throughput constraints over each CFDFC that was marked to be
  // optimized
  for (auto &[cfdfc, _] : vars.cfdfcs)
    if (funcInfo.cfdfcs[cfdfc])
      if (failed(addThroughputConstraints(*cfdfc)))
        return failure();

  // Finally, add the MILP objective
  return addObjective();
}

LogicalResult
BufferPlacementMILP::optimize(DenseMap<Value, PlacementResult> &placement) {
  if (unsatisfiable)
    return funcInfo.funcOp->emitError()
           << "The MILP is unsatisfiable: customized "
              "channel constraints are incompatible "
              "with buffers included inside units.";

  // Optimize the model, then check whether we found an optimal solution or
  // whether we reached the time limit

  if (logger)
    model.write(logger->getLogDir() + path::get_separator().str() +
                "placement_model.lp");
  model.optimize();
  if (logger)
    model.write(logger->getLogDir() + path::get_separator().str() +
                "placement_solutions.json");

  int status = model.get(GRB_IntAttr_Status);
  if (status != GRB_OPTIMAL && status != GRB_TIME_LIMIT)
    return funcInfo.funcOp->emitError()
           << "Gurobi failed with status code " << status;

  // Fill in placement information
  for (auto &[value, channelVars] : vars.channels) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    bool placeOpaque = channelVars.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;

    PlacementResult result;
    ChannelBufProps &props = channels[value];

    if (placeOpaque && numSlotsToPlace > 0) {
      /// NOTE: This matches the behavior of the legacy buffer placement pass
      /// However, a better placement may be achieved using the commented out
      /// logic below.
      // result.numTrans = props.minTrans;
      // result.numOpaque = numSlotsToPlace - props.minTrans;

      // We want as many slots as possible to be transparent and at least one
      // opaque slot, while satisfying all buffering constraints
      unsigned actualMinOpaque = std::max(1U, props.minOpaque);
      if (props.maxTrans.has_value() &&
          (props.maxTrans.value() < numSlotsToPlace - actualMinOpaque)) {
        result.numTrans = props.maxTrans.value();
        result.numOpaque = numSlotsToPlace - result.numTrans;
      } else {
        result.numOpaque = actualMinOpaque;
        result.numTrans = numSlotsToPlace - result.numOpaque;
      }
    } else
      // All slots should be transparent
      result.numTrans = numSlotsToPlace;

    Channel channel(value);
    deductInternalBuffers(channel, result);

    // If a single opaque slot is placed, add a transparent one behind it to
    // ensure circuit correctness
    if (result.numOpaque == 1 && result.numTrans == 0 &&
        props.maxTrans.value_or(1) > 0)
      result.numTrans = 1;

    placement[value] = result;
  }

  if (logger)
    logResults(placement);

  return success();
}

LogicalResult BufferPlacementMILP::createVars() {
  for (auto [idx, cfdfcAndOpt] : llvm::enumerate(funcInfo.cfdfcs))
    if (failed(createCFDFCVars(*cfdfcAndOpt.first, idx)))
      return failure();
  if (failed(createChannelVars()))
    return failure();

  // Update the model before returning so that these variables can be
  // referenced safely during the rest of model creation
  model.update();
  return success();
}

LogicalResult BufferPlacementMILP::createCFDFCVars(CFDFC &cfdfc, unsigned uid) {
  std::string prefix = "cfdfc" + std::to_string(uid) + "_";
  CFDFCVars cfdfcVars;

  // Create a continuous Gurobi variable of the given name
  auto createVar = [&](const std::string &name) {
    return model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
  };

  // Create a set of variables for each CFDFC unit
  for (auto [idx, unit] : llvm::enumerate(cfdfc.units)) {
    // Create the two unit variables
    UnitVars unitVar;
    std::string unitName = nameUniquer[unit] + std::to_string(idx);
    std::string varName = prefix + "inRetimeTok_" + unitName;
    unitVar.retIn = createVar(varName);

    // If the component is combinational (i.e., 0 latency) its output fluid
    // retiming equals its input fluid retiming, otherwise it is different
    double latency;
    if (failed(timingDB.getLatency(unit, latency)))
      latency = 0.0;
    if (latency == 0.0)
      unitVar.retOut = unitVar.retIn;
    else
      unitVar.retOut = createVar(prefix + "outRetimeTok_" + unitName);

    cfdfcVars.units[unit] = unitVar;
  }

  // Create a variable to represent the throughput of each CFDFC channel
  for (auto [idx, channel] : llvm::enumerate(cfdfc.channels))
    cfdfcVars.channelThroughputs[channel] =
        createVar(prefix + "throughput_" + getChannelName(channel));

  // Create a variable for the CFDFC's throughput
  cfdfcVars.throughput = createVar(prefix + "throughput");

  // Add the CFDFC variables to the global set of variables
  vars.cfdfcs[&cfdfc] = cfdfcVars;
  return success();
}

LogicalResult BufferPlacementMILP::createChannelVars() {
  // Create a set of variables for each channel in the circuit
  for (auto [idx, channelAndProps] : llvm::enumerate(channels)) {
    auto &channel = channelAndProps.first;

    // Construct a suffix for all variable names
    std::string suffix = "_" + getChannelName(channel);

    // Create a Gurobi variable of the given type and name
    auto createVar = [&](char type, const std::string &name) {
      return model.addVar(0, GRB_INFINITY, 0.0, type, name + suffix);
    };

    // Create the set of variables for the channel
    ChannelVars channelVars;
    channelVars.tPathIn = createVar(GRB_CONTINUOUS, "tPathIn");
    channelVars.tPathOut = createVar(GRB_CONTINUOUS, "tPathOut");
    channelVars.tElasIn = createVar(GRB_CONTINUOUS, "tElasIn");
    channelVars.tElasOut = createVar(GRB_CONTINUOUS, "tElasOut");
    channelVars.bufPresent = createVar(GRB_BINARY, "bufPresent");
    channelVars.bufIsOpaque = createVar(GRB_BINARY, "bufIsOpaque");
    channelVars.bufNumSlots = createVar(GRB_INTEGER, "bufNumSlots");

    vars.channels[channel] = channelVars;
  }
  return success();
}

LogicalResult
BufferPlacementMILP::addCustomChannelConstraints(ValueRange customChannels) {
  for (Value channel : customChannels) {
    ChannelVars &chVars = vars.channels[channel];
    ChannelBufProps &props = channels[channel];

    if (props.minOpaque > 0) {
      // Force the MILP to use opaque slots
      model.addConstr(chVars.bufIsOpaque == 1, "custom_forceOpaque");
      if (props.minTrans > 0) {
        // If the properties ask for both opaque and transaprent slots, let
        // opaque slots take over. Transparents slots will be placed
        // "manually" from the total number of slots indicated by the MILP's
        // result
        size_t idx;
        Operation *producer = getChannelProducer(channel, &idx);
        assert(producer && "channel producer must exist");
        producer->emitWarning()
            << "Outgoing channel " << idx
            << " requests placement of at least one transparent and at least "
               "one opaque slot on the channel, which the MILP does not "
               "formally support. To honor the properties, the MILP will be "
               "configured to only place opaque slots, some of which will be "
               "converted to transparent slots when parsing the MILP's "
               "solution.";
        unsigned minTotalSlots = props.minOpaque + props.minTrans;
        model.addConstr(chVars.bufNumSlots >= minTotalSlots,
                        "custom_minOpaqueAndTrans");
      } else
        // Force the MILP to place a minimum number of opaque slots
        model.addConstr(chVars.bufNumSlots >= props.minOpaque,
                        "custom_minOpaque");
    } else if (props.minTrans > 0)
      // Force the MILP to place a minimum number of transparent slots
      model.addConstr(chVars.bufNumSlots >= props.minTrans + chVars.bufIsOpaque,
                      "custom_minTrans");
    if (props.minOpaque + props.minTrans > 0)
      model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

    // Set a maximum number of slots to be placed
    if (props.maxOpaque.has_value()) {
      if (*props.maxOpaque == 0)
        // Force the MILP to use transparent slots
        model.addConstr(chVars.bufIsOpaque == 0, "custom_forceTransparent");
      if (props.maxTrans.has_value()) {
        // Force the MILP to use a maximum number of slots
        unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
        if (maxSlots == 0) {
          model.addConstr(chVars.bufPresent == 0, "custom_noBuffers");
          model.addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
        } else
          model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }
  }

  // Add constraints on paths
  for (auto [path, props] : pathConstraints) {
    // At most one channel gets a buffer on the path
    GRBLinExpr exprPresent;
    GRBLinExpr exprNum;
    GRBLinExpr exprOpaque;
    for (Value val : path) {
      exprPresent += vars.channels[val].bufPresent;
      exprNum += vars.channels[val].bufNumSlots;
      exprOpaque += vars.channels[val].bufIsOpaque;
    }
    model.addConstr(exprPresent <= 1, "custom_channel_path");

    if (props.minOpaque > 0) {
      // Force opaque buffers on the path
      for (Value val : path)
        model.addConstr(vars.channels[val].bufIsOpaque == 1,
                        "custom_path_forceOpaque");
    }

    if (unsigned numSlots = props.minOpaque + props.minTrans; numSlots > 0) {
      // Force a number of slots on the path
      if (props.minOpaque == 0)
        model.addConstr(exprNum >= numSlots, "custom_path_minOpaqueAndTrans");
      else
        model.addConstr(exprPresent >= numSlots + exprOpaque,
                        "custom_path_minTrans");
    }

    if (props.maxOpaque.has_value()) {
      if (*props.maxOpaque == 0) {
        // Force transparent slots only on the path
        for (Value val : path)
          model.addConstr(vars.channels[val].bufIsOpaque == 0,
                          "custom_path_forceTransparent");
      }
      if (props.maxTrans.has_value()) {
        // Force a maximum number of slots
        if (unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
            maxSlots == 0) {
          model.addConstr(exprPresent == 0, "custom_path_noBuffers");
          model.addConstr(exprNum == 0, "custom_path_noSlots");
        } else {
          model.addConstr(exprNum <= maxSlots, "custom_path_maxSlots");
        }
      }
    }
  }

  return success();
}

LogicalResult
BufferPlacementMILP::addPathConstraints(ValueRange pathChannels,
                                        ArrayRef<Operation *> pathUnits) {
  // Add path constraints for channels
  for (Value channel : pathChannels) {
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &t1 = chVars.tPathIn;
    GRBVar &t2 = chVars.tPathOut;
    // Arrival time at channel's input must be lower than target clock period
    model.addConstr(t1 <= targetPeriod, "path_channelInPeriod");
    // Arrival time at channel's output must be lower than target clock period
    model.addConstr(t2 <= targetPeriod, "path_channelOutPeriod");
    // If there isn't an opaque buffer on the channel, arrival time at
    // channel's output must be greater than at channel's input. We also need
    // to account for any channel delay
    double delay = channels[channel].delay;
    model.addConstr(t2 >= t1 + delay - maxPeriod * chVars.bufIsOpaque,
                    "path_opaqueChannel");
  }

  // Add path constraints for units
  for (Operation *op : pathUnits) {
    double latency;
    if (failed(timingDB.getLatency(op, latency)))
      latency = 0.0;

    if (latency == 0.0) {
      double dataDelay;
      if (failed(timingDB.getTotalDelay(op, SignalType::DATA, dataDelay)))
        dataDelay = 0.0;

      // The unit is not pipelined, add a path constraint for each
      // input/output port pair in the unit
      forEachIOPair(op, [&](Value in, Value out) {
        GRBVar &tInPort = vars.channels[in].tPathOut;
        GRBVar &tOutPort = vars.channels[out].tPathIn;
        // Arrival time at unit's output port must be greater than arrival
        // time at unit's input port + the unit's combinational data delay
        model.addConstr(tOutPort >= tInPort + dataDelay, "path_combDelay");
      });
    } else {
      // The unit is pipelined, add a constraint for every of the unit's
      // inputs and every of the unit's output ports

      // Input port constraints
      for (Value inChannel : op->getOperands()) {
        if (!vars.channels.contains(inChannel))
          continue;

        double inPortDelay;
        if (failed(timingDB.getPortDelay(op, SignalType::DATA, PortType::IN,
                                         inPortDelay)))
          inPortDelay = 0.0;

        GRBVar &tInPort = vars.channels[inChannel].tPathOut;
        // Arrival time at unit's input port + input port delay must be less
        // than the target clock period
        model.addConstr(tInPort + inPortDelay <= targetPeriod, "path_inDelay");
      }

      // Output port constraints
      for (OpResult outChannel : op->getResults()) {
        if (!vars.channels.contains(outChannel))
          continue;

        double outPortDelay;
        if (failed(timingDB.getPortDelay(op, SignalType::DATA, PortType::OUT,
                                         outPortDelay)))
          outPortDelay = 0.0;

        GRBVar &tOutPort = vars.channels[outChannel].tPathIn;
        // Arrival time at unit's output port is equal to the output port
        // delay
        model.addConstr(tOutPort == outPortDelay, "path_outDelay");
      }
    }
  }
  return success();
}

LogicalResult BufferPlacementMILP::addElasticityConstraints(
    ValueRange elasticChannels, ArrayRef<Operation *> elasticUnits) {
  // Upper bound for the longest rigid path
  unsigned cstCoef = std::distance(funcInfo.funcOp.getOps().begin(),
                                   funcInfo.funcOp.getOps().end()) +
                     2;

  // Add elasticity constraints for channels
  for (Value channel : elasticChannels) {
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &tIn = chVars.tElasIn;
    GRBVar &tOut = chVars.tElasOut;
    GRBVar &present = chVars.bufPresent;
    GRBVar &opaque = chVars.bufIsOpaque;
    GRBVar &numSlots = chVars.bufNumSlots;

    // If there is an opaque buffer on the channel, the channel elastic
    // arrival time at the ouput must be greater than at the input (breaks
    // cycles!)
    model.addConstr(tOut >= tIn - cstCoef * opaque, "elastic_cycle");
    // If there is an opaque buffer, there must be at least one slot
    model.addConstr(numSlots >= opaque, "elastic_slots");
    // If there is at least one slot, there must be a buffer
    model.addConstr(present >= 0.01 * numSlots, "elastic_present");
  }

  // Add an elasticity constraint for every input/output port pair in the
  // elastic units
  for (Operation *op : elasticUnits) {
    forEachIOPair(op, [&](Value in, Value out) {
      GRBVar &tInPort = vars.channels[in].tElasOut;
      GRBVar &tOutPort = vars.channels[out].tElasIn;
      // The elastic arrival time at the output port must be at least one
      // greater than at the input port
      model.addConstr(tOutPort >= 1 + tInPort, "elastic_unitTime");
    });
  }
  return success();
}

LogicalResult BufferPlacementMILP::addThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfdfcVars = vars.cfdfcs[&cfdfc];

  // Add a set of constraints for each CFDFC channel
  for (auto &[channel, chThroughput] : cfdfcVars.channelThroughputs) {
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *(channel.getUsers().begin());

    assert(vars.channels.contains(channel) && "unknown channel");
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &retSrc = cfdfcVars.units[srcOp].retOut;
    GRBVar &retDst = cfdfcVars.units[dstOp].retIn;
    GRBVar &bufIsOpaque = chVars.bufIsOpaque;
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &throughput = cfdfcVars.throughput;
    unsigned backedge = cfdfc.backedges.contains(channel) ? 1 : 0;

    /// TODO: The legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed. Temporarily, emulate the same behavior obtained from passing
    /// our DOTs to the old buffer pass by assuming the "true" input is always
    /// the least executed one
    if (arith::SelectOp selOp = dyn_cast<arith::SelectOp>(dstOp))
      if (channel == selOp.getTrueValue())
        continue;

    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chThroughput - backedge == retDst - retSrc,
                    "throughput_channelRetiming");
    // If there is an opaque buffer, the CFDFC throughput cannot exceed the
    // channel throughput. If there is not, the CFDFC throughput can exceed
    // the channel thoughput by 1
    model.addConstr(throughput - chThroughput + bufIsOpaque <= 1,
                    "throughput_cfdfc");
    // If there is an opaque buffer, the summed channel and CFDFC throughputs
    // cannot exceed the number of buffer slots. If there is not, the combined
    // throughput can exceed the number of slots by 1
    model.addConstr(chThroughput + throughput + bufIsOpaque - bufNumSlots <= 1,
                    "throughput_combined");
    // The channel's throughput cannot exceed the number of buffer slots
    model.addConstr(chThroughput <= bufNumSlots, "throughput_channel");
  }

  // Add a constraint for each pipelined CFDFC unit
  for (auto &[op, unitVars] : cfdfcVars.units) {
    double latency;
    if (failed(timingDB.getLatency(op, latency)) || latency == 0.0)
      continue;

    GRBVar &retIn = unitVars.retIn;
    GRBVar &retOut = unitVars.retOut;
    GRBVar &throughput = cfdfcVars.throughput;
    // The fluid retiming of tokens across the non-combinational unit must
    // be the same as its latency multiplied by the CFDFC's throughput
    model.addConstr(throughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }
  return success();
}

LogicalResult BufferPlacementMILP::addObjective() {
  // Compute the total number of executions over all channels
  unsigned totalExecs = 0;
  for (auto &[channel, _] : vars.channels)
    totalExecs += getChannelNumExecs(channel);

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each CFDFC, add a throughput contribution to the objective, weighted
  // by the "importance" of the CFDFC
  double maxCoefCFDFC = 0.0;
  if (totalExecs != 0) {
    for (auto &[cfdfc, cfdfcVars] : vars.cfdfcs) {
      if (!funcInfo.cfdfcs[cfdfc])
        continue;
      double coef = cfdfc->channels.size() * cfdfc->numExecs /
                    static_cast<double>(totalExecs);
      objective += coef * cfdfcVars.throughput;
      maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
    }
  }

  // In case we ran the MILP without providing any CFDFC, set the maximum
  // CFDFC coefficient to any positive value
  if (maxCoefCFDFC == 0.0)
    maxCoefCFDFC = 1.0;

  // For each channel, add a "penalty" in case a buffer is added to the
  // channel, and another penalty that depends on the number of slots
  double bufPenaltyMul = 1e-4;
  double slotPenaltyMul = 1e-5;
  for (auto &[channel, chVar] : vars.channels) {
    objective -= maxCoefCFDFC * bufPenaltyMul * chVar.bufPresent;
    objective -= maxCoefCFDFC * slotPenaltyMul * chVar.bufNumSlots;
  }

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
  return success();
}

LogicalResult BufferPlacementMILP::addInternalBuffers(Channel &channel) {
  // Add slots present at the source unit's output ports
  std::string srcName = channel.producer.getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(&channel.producer)) {
    channel.props->minTrans += model->outputModel.transparentSlots;
    channel.props->minOpaque += model->outputModel.opaqueSlots;
  }

  // Add slots present at the destination unit's input ports
  std::string dstName = channel.consumer.getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(&channel.consumer)) {
    channel.props->minTrans += model->inputModel.transparentSlots;
    channel.props->minOpaque += model->inputModel.opaqueSlots;
  }

  return success(channel.props->isSatisfiable());
}

void BufferPlacementMILP::deductInternalBuffers(Channel &channel,
                                                PlacementResult &result) {
  std::string srcName = channel.producer.getName().getStringRef().str();
  std::string dstName = channel.consumer.getName().getStringRef().str();
  unsigned numTransToDeduct = 0, numOpaqueToDeduct = 0;

  // Remove slots present at the source unit's output ports
  if (const TimingModel *model = timingDB.getModel(&channel.producer)) {
    numTransToDeduct += model->outputModel.transparentSlots;
    numOpaqueToDeduct += model->outputModel.opaqueSlots;
  }
  // Remove slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(&channel.consumer)) {
    numTransToDeduct += model->inputModel.transparentSlots;
    numOpaqueToDeduct += model->inputModel.opaqueSlots;
  }

  assert(result.numTrans >= numTransToDeduct &&
         "not enough transparent slots were placed, the MILP was likely "
         "incorrectly configured");
  assert(result.numOpaque >= numOpaqueToDeduct &&
         "not enough opaque slots were placed, the MILP was likely "
         "incorrectly configured");
  result.numTrans -= numTransToDeduct;
  result.numOpaque -= numOpaqueToDeduct;
}

std::string BufferPlacementMILP::getChannelName(Value channel) {
  Operation *consumer = *channel.getUsers().begin();
  if (BlockArgument arg = dyn_cast<BlockArgument>(channel)) {
    return "arg" + std::to_string(arg.getArgNumber()) + "_" +
           nameUniquer[consumer];
  }
  OpResult res = dyn_cast<OpResult>(channel);
  return nameUniquer[res.getDefiningOp()] + "_" +
         std::to_string(res.getResultNumber()) + "_" + nameUniquer[consumer];
}

void BufferPlacementMILP::forEachIOPair(
    Operation *op, const std::function<void(Value, Value)> &callback) {
  for (Value opr : op->getOperands())
    if (!isa<MemRefType>(opr.getType()))
      for (OpResult res : op->getResults())
        if (!isa<MemRefType>(res.getType()))
          callback(opr, res);
}

unsigned BufferPlacementMILP::getChannelNumExecs(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  if (!srcOp)
    // A channel which originates from a function argument executes only once
    return 1;

  // Iterate over all CFDFCs which contain the channel to determine its total
  // number of executions. Backedges are executed one less time than "forward
  // edges" since they are only taken between executions of the cycle the
  // CFDFC represents
  unsigned numExec = isBackedge(channel) ? 0 : 1;
  for (auto &[cfdfc, _] : funcInfo.cfdfcs)
    if (cfdfc->channels.contains(channel))
      numExec += cfdfc->numExecs;
  return numExec;
}

void BufferPlacementMILP::logResults(
    const DenseMap<Value, PlacementResult> &placement) {
  assert(logger && "no logger was provided");
  mlir::raw_indented_ostream &os = **logger;

  os << "# ========================== #\n";
  os << "# Buffer Placement Decisions #\n";
  os << "# ========================== #\n\n";

  for (auto &[value, channelVars] : vars.channels) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    // Extract number and type of slots
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    bool placeOpaque = channelVars.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;

    PlacementResult result;
    ChannelBufProps &props = channels[value];

    // Log placement decision
    os << getChannelName(value) << ":\n";
    os.indent();
    std::stringstream propsStr;
    propsStr << props;
    os << "- Buffering constraints: " << propsStr.str() << "\n";
    os << "- MILP decision: " << numSlotsToPlace << " "
       << (placeOpaque ? "opaque" : "transparent") << " slot(s)\n";
    os << "- Placement decision: " << result.numTrans
       << " transparent slot(s) and " << result.numOpaque
       << " opaque slot(s)\n";
    os.unindent();
    os << "\n";
  }

  os << "# ================= #\n";
  os << "# CFDFC Throughputs #\n";
  os << "# ================= #\n\n";

  // Log global CFDFC throuhgputs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcs)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
    os << "Throughput of CFDFC #" << idx << ": " << throughput << "\n";
  }

  os << "\n# =================== #\n";
  os << "# Channel Throughputs #\n";
  os << "# =================== #\n\n";

  // Log throughput of all channels in all CFDFCs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcs)) {
    auto [cf, cfVars] = cfdfcWithVars;
    os << "Per-channel throughputs of CFDFC #" << idx << ":\n";
    os.indent();
    for (auto [val, channelTh] : cfVars.channelThroughputs) {
      os << getChannelName(val) << ": " << channelTh.get(GRB_DoubleAttr_X)
         << "\n";
    }
    os.unindent();
    os << "\n";
  }
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
