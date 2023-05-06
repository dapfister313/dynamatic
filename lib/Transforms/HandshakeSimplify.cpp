//===- HandshakeSimplify.h - Simplifies stuff -------------------*- C++ -*-===//
//
// This file declares the --handshake-simplify pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeSimplify.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

namespace {

struct EraseUnconditionalBranches
    : public OpConversionPattern<handshake::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::BranchOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Just delete it
    rewriter.updateRootInPlace(op, [&] {
      op.getDataResult().replaceAllUsesWith(op.getDataOperand());
      rewriter.eraseOp(op);
    });
    return success();
  }
};

struct EraseSimpleMerges : public OpConversionPattern<handshake::MergeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::MergeOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1)
      // Just delete it
      rewriter.updateRootInPlace(op, [&] {
        op.getResult().replaceAllUsesWith(op.getDataOperands().front());
        rewriter.eraseOp(op);
      });
    return success();
  }
};

struct HandshakeSimplifyPass
    : public HandshakeSimplifyBase<HandshakeSimplifyPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    ConversionTarget target(*ctx);
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalOp<handshake::BranchOp>();
    target.addDynamicallyLegalOp<handshake::MergeOp>(
        [&](const auto &op) { return op->getNumOperands() != 1; });

    patterns.add<EraseUnconditionalBranches, EraseSimpleMerges>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

namespace dynamatic {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createHandshakeSimplify() {
  return std::make_unique<HandshakeSimplifyPass>();
}
} // namespace dynamatic
