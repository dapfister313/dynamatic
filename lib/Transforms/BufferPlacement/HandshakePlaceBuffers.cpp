//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffer placement in Handshake functions, using a set of available algorithms
// (all of which currently require Gurobi to solve MILPs). Buffers are placed to
// ensure circuit correctness and increase performance.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <string>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;
namespace {

/// Thin wrapper around a `Logger` that allows to conditionally create the
/// resources to write to the file based on a flag provided during objet
/// creation. This enables us to use RAII to deallocate the logger only if it
/// was created.
class BufferLogger {
public:
  /// The underlying logger object, which may remain nullptr.
  Logger *log = nullptr;

  /// Optionally allocates a logger based on whether the `dumpLogs` flag is set.
  /// If it is, the log file's location is determined based om the provided
  /// function's name. On error, `ec` will contain a non-zero error code
  /// and the logger should not be used.
  BufferLogger(handshake::FuncOp funcOp, bool dumpLogs, std::error_code &ec);

  /// Returns the underlying logger, which may be nullptr.
  Logger *operator*() { return log; }

  /// Returns the underlying indented writer stream to the log file. Requires
  /// the object to have been created with the `dumpLogs` flag set to true.
  mlir::raw_indented_ostream &getStream() {
    assert(log && "logger was not allocated");
    return **log;
  }

  BufferLogger(const BufferLogger *) = delete;
  BufferLogger operator=(const BufferLogger *) = delete;

  /// Deletes the underlying logger object if it was allocated.
  ~BufferLogger() {
    if (log)
      delete log;
  }
};
} // namespace

BufferLogger::BufferLogger(handshake::FuncOp funcOp, bool dumpLogs,
                           std::error_code &ec) {
  if (!dumpLogs)
    return;

  std::string sep = path::get_separator().str();
  std::string fp = "buffer-placement" + sep + funcOp.getName().str() + sep;
  log = new Logger(fp + "placement.log", ec);
}

HandshakePlaceBuffersPass::HandshakePlaceBuffersPass(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  this->algorithm = algorithm.str();
  this->frequencies = frequencies.str();
  this->timingModels = timingModels.str();
  this->firstCFDFC = firstCFDFC;
  this->targetCP = targetCP;
  this->timeout = timeout;
  this->dumpLogs = dumpLogs;
}

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
void HandshakePlaceBuffersPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  modOp.emitError() << "Project was built without Gurobi installed, can't "
                       "run smart buffer placement pass\n";
  return signalPassFailure();
}
#else
/// Logs arch and CFDFC information (sequence of basic blocks, number of
/// executions, channels, units) to the logger.
static void logFuncInfo(FuncInfo &info, Logger &log) {
  mlir::raw_indented_ostream &os = *log;
  os << "# ===== #\n";
  os << "# Archs #\n";
  os << "# ===== #\n\n";

  for (ArchBB &arch : info.archs) {
    os << arch.srcBB << " -> " << arch.dstBB << " with " << arch.numTrans
       << " executions";
    if (arch.isBackEdge)
      os << " (backedge)";
    os << "\n";
  }

  os << "\n# ================ #\n";
  os << "# Extracted CFDFCs #\n";
  os << "# ================ #\n\n";

  for (auto [idx, cfAndOpt] : llvm::enumerate(info.cfdfcs)) {
    auto &[cf, _] = cfAndOpt;
    os << "CFDFC #" << idx << ": ";
    for (size_t i = 0, e = cf->cycle.size() - 1; i < e; ++i)
      os << cf->cycle[i] << " -> ";
    os << cf->cycle.back() << "\n";
    os.indent();
    os << "- Number of executions: " << cf->numExecs << "\n";
    os << "- Number of units: " << cf->units.size() << "\n";
    os << "- Number of channels: " << cf->channels.size() << "\n";
    os << "- Number of backedges: " << cf->backedges.size() << "\n\n";
    os.unindent();
  }

  os.flush();
}

void HandshakePlaceBuffersPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  // Check that the algorithm exists
  if (algorithm != "fpga20" && algorithm != "fpga20-legacy") {
    modOp->emitError() << "Unknown algorithm '" << algorithm
                       << "', possible choices are 'fpga20', 'fpga20-legacy'.";
    return signalPassFailure();
  }

  // Check IR invariants and parse basic block archs from disk
  DenseMap<handshake::FuncOp, FuncInfo> funcToInfo;
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    funcToInfo.insert(std::make_pair(funcOp, FuncInfo(funcOp)));
    FuncInfo &info = funcToInfo[funcOp];

    // Read the CSV containing arch information (number of transitions between
    // pairs of basic blocks) from disk. While the rest of this pass works if
    // the module contains multiple functions, this only makes sense if the
    // module has a single function
    if (failed(StdProfiler::readCSV(frequencies, info.archs))) {
      funcOp->emitError() << "Failed to read profiling information from CSV";
      return signalPassFailure();
    }

    if (failed(checkFuncInvariants(info)))
      return signalPassFailure();
  }

  // Read the operations' timing models from disk
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();

  // Place buffers in each function
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
    if (failed(placeBuffers(funcToInfo[funcOp], timingDB)))
      return signalPassFailure();
}

LogicalResult HandshakePlaceBuffersPass::checkFuncInvariants(FuncInfo &info) {
  handshake::FuncOp funcOp = info.funcOp;

  // Verify that the IR is in a valid state for buffer placement
  // Buffer placement requires that all values are used exactly once
  if (failed(verifyAllValuesHasOneUse(funcOp)))
    return funcOp.emitOpError() << "Not all values are used exactly once";

  // Perform a number of verifications to make sure that the Handshake function
  // whose information is passed as argument is valid for buffer placement

  // Store all archs in a map for fast query time
  DenseMap<unsigned, llvm::SmallDenseSet<unsigned, 2>> transitions;
  for (ArchBB &arch : info.archs)
    transitions[arch.srcBB].insert(arch.dstBB);

  // Store the BB to which each block belongs for quick access later
  DenseMap<Operation *, std::optional<unsigned>> opBlocks;
  for (Operation &op : info.funcOp.getOps())
    opBlocks[&op] = getLogicBB(&op);

  for (Operation &op : info.funcOp.getOps()) {
    // Most operations should belong to a basic block for buffer placement to
    // work correctly. Don't outright fail in case one operation is outside of
    // all blocks but warn the user
    if (!isa<handshake::SinkOp, handshake::MemoryControllerOp>(&op))
      if (!getLogicBB(&op).has_value())
        op.emitWarning() << "Operation does not belong to any block, MILP "
                            "behavior may be suboptimal or incorrect.";

    std::optional<unsigned> srcBB = opBlocks[&op];
    for (OpResult res : op.getResults()) {
      Operation *user = *res.getUsers().begin();
      std::optional<unsigned> dstBB = opBlocks[user];

      // All transitions between blocks must exist in the original CFG
      if (srcBB.has_value() && dstBB.has_value() && *srcBB != *dstBB &&
          !transitions[*srcBB].contains(*dstBB))
        return op.emitError()
               << "Result " << res.getResultNumber() << " defined in block "
               << *srcBB << " is used in block " << *dstBB
               << ". This connection does not exist according to the CFG "
                  "graph. Solving the buffer placement MILP would yield an "
                  "incorrect placement.";
    }
  }
  return success();
}

LogicalResult
HandshakePlaceBuffersPass::placeBuffers(FuncInfo &info,
                                        TimingDatabase &timingDB) {
  // Use a wrapper around a logger to benefit from RAII
  std::error_code ec;
  BufferLogger bufLogger(info.funcOp, dumpLogs, ec);
  if (ec.value() != 0) {
    info.funcOp->emitError() << "Failed to create logger for function "
                             << info.funcOp.getName() << "\n"
                             << ec.message();
    return failure();
  }
  Logger *logger = dumpLogs ? *bufLogger : nullptr;

  // Get CFDFCs from the function unless the functions has no archs (i.e.,
  // it has a single block) in which case there are no CFDFCs
  SmallVector<CFDFC> cfdfcs;
  if (!info.archs.empty() && failed(getCFDFCs(info, logger, cfdfcs)))
    return failure();

  // All extracted CFDFCs must be optimized
  for (CFDFC &cf : cfdfcs)
    info.cfdfcs[&cf] = true;

  if (dumpLogs)
    logFuncInfo(info, *logger);

  // Solve the MILP to obtain a buffer placement
  DenseMap<Value, PlacementResult> placement;
  if (failed(getBufferPlacement(info, timingDB, logger, placement)))
    return failure();

  instantiateBuffers(placement);
  return success();
}

LogicalResult HandshakePlaceBuffersPass::getCFDFCs(FuncInfo &info,
                                                   Logger *logger,
                                                   SmallVector<CFDFC> &cfdfcs) {
  SmallVector<ArchBB> archsCopy(info.archs);

  // Store all archs in a set. We use a pointer to each arch as the key type to
  // allow us to modify their frequencies during CFDFC extractions without
  // messing up key hashes
  ArchSet archs;
  // Similarly, store all block IDs in a set.
  BBSet bbs;
  for (ArchBB &arch : archsCopy) {
    archs.insert(&arch);
    bbs.insert(arch.srcBB);
    bbs.insert(arch.dstBB);
  }

  // Set of selected archs
  ArchSet selectedArchs;
  // Number of executions
  unsigned numExecs;
  do {
    // Clear the sets of selected archs and BBs
    selectedArchs.clear();

    // Path where to dump the MILP model and solutions, if necessary
    std::string logPath = "";
    if (logger)
      logPath = logger->getLogDir() + path::get_separator().str() + "cfdfc" +
                std::to_string(cfdfcs.size());

    // Try to extract the next CFDFC
    int milpStat;
    if (failed(extractCFDFC(info.funcOp, archs, bbs, selectedArchs, numExecs,
                            logPath, &milpStat)))
      return info.funcOp->emitError()
             << "CFDFC extraction MILP failed with status " << milpStat << ". "
             << getGurobiOptStatusDesc(milpStat);
    if (numExecs == 0)
      break;

    // Create the CFDFC from the set of selected archs and BBs
    cfdfcs.emplace_back(info.funcOp, selectedArchs, numExecs);
  } while (!firstCFDFC);

  return success();
}

LogicalResult HandshakePlaceBuffersPass::getBufferPlacement(
    FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
    DenseMap<Value, PlacementResult> &placement) {

  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  // Create and solve the MILP
  BufferPlacementMILP *milp = nullptr;
  if (algorithm == "fpga20")
    milp = new fpga20::FPGA20Buffers(info, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, false);
  else if (algorithm == "fpga20-legacy")
    milp = new fpga20::FPGA20Buffers(info, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, true);
  assert(milp && "unknown placement algorithm");
  int milpStat;
  LogicalResult res = success();
  if (failed(milp->optimize(&milpStat))) {
    res = info.funcOp->emitError()
          << "Buffer placement MILP failed with status " << milpStat
          << ", reason:" << getGurobiOptStatusDesc(milpStat);
  } else if (failed(milp->getPlacement(placement))) {
    res = info.funcOp->emitError()
          << "Failed to extract placement decisions from MILP's solution.";
  }

  delete milp;
  return res;
}

void HandshakePlaceBuffersPass::instantiateBuffers(
    DenseMap<Value, PlacementResult> &placement) {
  OpBuilder builder(&getContext());
  for (auto &[channel, placeRes] : placement) {
    Operation *opSrc = channel.getDefiningOp();
    Operation *opDst = *channel.getUsers().begin();
    builder.setInsertionPointAfter(opSrc);

    Value bufferIn = channel;
    auto placeBuffer = [&](BufferTypeEnum bufType, unsigned numSlots) {
      if (numSlots == 0)
        return;

      // Insert an opaque buffer
      auto bufOp = builder.create<handshake::BufferOp>(
          bufferIn.getLoc(), bufferIn, numSlots, bufType);
      inheritBB(opSrc, bufOp);
      Value bufferRes = bufOp.getResult();

      opDst->replaceUsesOfWith(bufferIn, bufferRes);
      bufferIn = bufferRes;
    };

    if (placeRes.opaqueBeforeTrans) {
      placeBuffer(BufferTypeEnum::seq, placeRes.numOpaque);
      placeBuffer(BufferTypeEnum::fifo, placeRes.numTrans);
    } else {
      placeBuffer(BufferTypeEnum::fifo, placeRes.numTrans);
      placeBuffer(BufferTypeEnum::seq, placeRes.numOpaque);
    }
  }
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

std::string dynamatic::buffer::getGurobiOptStatusDesc(int status) {
  switch (status) {
  case 1:
    return "Model is loaded, but no solution information is available.";
  case 2:
    return "Model was solved to optimality (subject to tolerances), and an "
           "optimal solution is available.";
  case 3:
    return "Model was proven to be infeasible.";
  case 4:
    return "Model was proven to be either infeasible or unbounded. To obtain a "
           "more definitive conclusion, set the DualReductions parameter to 0 "
           "and reoptimize.";
  case 5:
    return "Model was proven to be unbounded.";
  case 9:
    return "Optimization terminated because the time expended exceeded the "
           "value specified in the TimeLimit parameter.";
  default:
    return "No description available for optimization status code " +
           std::to_string(status) +
           ". Check "
           "https://www.gurobi.com/documentation/current/refman/"
           "optimization_status_codes.html for more details";
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakePlaceBuffers(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<HandshakePlaceBuffersPass>(
      algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs);
}
