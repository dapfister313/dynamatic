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

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_ITERATIVEBUFFERS_H
