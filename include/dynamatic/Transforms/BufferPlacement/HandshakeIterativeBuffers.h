//===- HandshakeIterativeBuffers.h - Iter. buffer placement -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-iterative-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_ITERATIVEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_ITERATIVEBUFFERS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <list>
#include <map>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
/*
namespace dynamatic {
namespace buffer {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeIterativeBuffers(StringRef algorithm = "fpga20",
                                StringRef frequencies = "",
                                StringRef timingModels = "",
                                bool firstCFDFC = false, double targetCP = 4.0,
                                unsigned timeout = 180, bool dumpLogs = false);

#define GEN_PASS_DECL_HANDSHAKEITERATIVEBUFFERS
#define GEN_PASS_DEF_HANDSHAKEITERATIVEBUFFERS
#include "dynamatic/Transforms/Passes.h.inc"

/// Stores some data you may want to extract from buffer placement
struct MyData {
  //extracts needed resource sharing data from FuncInfo struct
  ResourceSharingInfo sharing_feedback; 
  SmallVector<experimental::ArchBB> archs;
  unsigned someCountOfSomething = 0;
  unsigned totalNumberOfOpaqueBuffers = 0;
};

/// Sub-type of the classic buffer placement pass, just so that we can override
/// some of the methods used during buffer placement and extract internal data
/// (e.g., channel throughputs as determined by the MILP) from the pass.
struct MyBufferPlacementPass : public HandshakePlaceBuffersPass {
  MyBufferPlacementPass(MyData &data, StringRef algorithm,
                        StringRef frequencies, StringRef timingModels,
                        bool firstCFDFC, double targetCP, unsigned timeout,
                        bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        data(data){};

  /// Some data you care about extracting.
  MyData &data;

protected:
  /// Custom buffer placement step. Copied from `HandshakePlaceBuffersPass` with
  /// the addition of a step at the end to extract some information from the
  /// MILP.
  LogicalResult
  getBufferPlacement(FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
                     DenseMap<Value, PlacementResult> &placement) override;
};

} // namespace buffer
} // namespace dynamatic
*/
#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_ITERATIVEBUFFERS_H