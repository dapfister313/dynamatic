//===- SimpleTransform.cpp - Simple IR transformation pass ------*- C++ -*-===//
//
// Implements the --dynamatic-tutorial-simple-transform pass.
//
//===----------------------------------------------------------------------===//

#include "CreatingPasses/Transforms/SimpleTransform.h"
#include "CreatingPasses/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace dynamatic::tutorials;
using namespace circt::handshake;

namespace {

struct SimpleTransformPass : public SimpleTransformBase<SimpleTransformPass> {

  void runOnOperation() override { auto *ctx = &getContext(); };
};
} // namespace

namespace dynamatic {
namespace tutorials {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createSimpleTransformPass() {
  return std::make_unique<SimpleTransformPass>();
}
} // namespace tutorials
} // namespace dynamatic
