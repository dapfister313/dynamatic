//===- HandshakeRemoveMemories.h - Match argument names with C --00-*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-remove-memories pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_HANDHSHAKEREMOVEMEMORIES_H
#define EXPERIMENTAL_TRANSFORMS_HANDHSHAKEREMOVEMEMORIES_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace experimental {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeRemoveMemories();

#define GEN_PASS_DECL_HANDSHAKEREMOVEMEMORIES
#define GEN_PASS_DEF_HANDSHAKEREMOVEMEMORIES
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TRANSFORMS_HANDHSHAKEREMOVEMEMORIES_H
