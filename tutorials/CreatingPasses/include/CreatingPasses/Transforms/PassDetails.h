//===- PassDetails.h - Pass classes -----------------------------*- C++ -*-===//
//
// It contains forward declarations needed by tranformation passes and includes
// auto-generated base class definitions for all tranformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CREATINGAPASS_TRANSFORMS_PASSDETAILS_H
#define CREATINGAPASS_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace handshake {
class HandshakeDialect;
} // namespace handshake
} // namespace circt

namespace dynamatic {
namespace tutorials {
#define GEN_PASS_CLASSES
#include "CreatingPasses/Transforms/Passes.h.inc"
} // namespace tutorials
} // namespace dynamatic

#endif // CREATINGAPASS_TRANSFORMS_PASSDETAILS_H
