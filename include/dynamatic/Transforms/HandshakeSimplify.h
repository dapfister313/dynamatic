//===- HandshakeSimplify.h - Simplifies stuff -------------------*- C++ -*-===//
//
// This file declares the --handshake-simplify pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKESIMPLIFY_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKESIMPLIFY_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createHandshakeSimplify();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKESIMPLIFY_H
