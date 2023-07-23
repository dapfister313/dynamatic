//===- HandshakeOptimizeBitwidths.cpp - Optimize channel widths -*- C++ -*-===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeOptimizeBitwidths.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;

static bool isIntType(Type type) { return isa<IntegerType>(type); }

static bool isBitwidthMod(Operation *op) {
  return isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op);
}

static void dematerializeBitwidthMods(handshake::FuncOp funcOp,
                                      MLIRContext *ctx) {
  OpBuilder builder(ctx);
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    if (isBitwidthMod(&op)) {
      op.getResult(0).replaceAllUsesWith(op.getOperand(0));
      op.erase();
    }
  }
}

static void addModsForValue(Value val, Location loc, Builder &builder) {
  // Type resType = val.getType();
  // if (!isa<IntegerType, FloatType>(resType))
  //   return;

  // SmallDenseMap<unsigned, Operation *, 4> mods;
  // for (OpOperand &use : val.getUses()) {
  //   if (use.value)
  // }
}

static void materializeBitwidthMods(handshake::FuncOp funcOp,
                                    MLIRContext *ctx) {
  OpBuilder builder(ctx);
  // Materialize bitwidths modifiers for function arguments
  Block &entryBlock = funcOp.front();
  builder.setInsertionPointToStart(&entryBlock);
  for (auto arg : entryBlock.getArguments())
    addModsForValue(arg, funcOp->getLoc(), builder);

  // Materialize bitwidths modifiers for all operations in the function body
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    builder.setInsertionPointToStart(&entryBlock);
    for (Value result : op.getResults())
      addModsForValue(result, op.getLoc(), builder);
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
static Value getMinimalValue(Value val) {
  Type type = val.getType();
  assert(isIntType(type) && "value must be integer type");

  Operation *defOp = val.getDefiningOp();
  if (!defOp || !isa<arith::ExtSIOp, arith::ExtUIOp>(defOp))
    return val;

  return getMinimalValue(defOp->getOperand(0));
}

static unsigned getActualResultWidth(Value val) {
  Type resType = val.getType();
  assert(isIntType(resType) && "value must be integer type");

  // Find the value use that discards the least amount of bits. This gives us
  // the amount of bits of value that can be safely discarded
  std::optional<unsigned> maxWidth;
  for (Operation *user : val.getUsers()) {
    if (!isa<arith::TruncIOp>(user))
      return resType.getIntOrFloatBitWidth();
    unsigned truncWidth = user->getResult(0).getType().getIntOrFloatBitWidth();
    maxWidth = std::max(maxWidth.value_or(0), truncWidth);
  }

  return maxWidth.value_or(resType.getIntOrFloatBitWidth());
}

static Operation *extendValue(Value val, unsigned extWidth,
                              PatternRewriter &rewriter) {
  Type type = val.getType();
  assert(isIntType(type) && "value must be integer type");

  return rewriter.create<arith::ExtSIOp>(
      val.getLoc(), rewriter.getIntegerType(extWidth), val);
}

static Operation *truncateValue(Value val, unsigned truncWidth,
                                PatternRewriter &rewriter) {
  Type type = val.getType();
  assert(isIntType(type) && "value must be integer type");

  return rewriter.create<arith::TruncIOp>(
      val.getLoc(), rewriter.getIntegerType(truncWidth), val);
}

static Value modifyBitwidth(Value val, unsigned requiredWidth,
                            PatternRewriter &rewriter) {
  Type type = val.getType();
  assert(isIntType(type) && "value must be integer type");

  unsigned width = type.getIntOrFloatBitWidth();
  if (width < requiredWidth)
    return rewriter
        .create<arith::ExtSIOp>(val.getLoc(),
                                rewriter.getIntegerType(requiredWidth), val)
        .getResult();
  if (width > requiredWidth)
    return rewriter
        .create<arith::TruncIOp>(val.getLoc(),
                                 rewriter.getIntegerType(requiredWidth), val)
        .getResult();
  return val;
}

namespace {

template <typename ModOp>
struct EraseUselessBitwidthModifier : public OpRewritePattern<ModOp> {
  using OpRewritePattern<ModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModOp modOp,
                                PatternRewriter &rewriter) const override {
    if (modOp->getResult(0).getType() != modOp->getOperand(0).getType())
      return failure();

    rewriter.replaceOp(modOp, modOp->getOperand(0));
    return success();
  }
};

template <typename ModOp>
struct CombineBitwidthModifiers : public OpRewritePattern<ModOp> {
  using OpRewritePattern<ModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModOp modOp,
                                PatternRewriter &rewriter) const override {
    // Modifier must have a single user
    if (!modOp->getResult(0).hasOneUse())
      return failure();

    // Its user must be of the same type as it is
    Operation *modUser = *modOp->getUsers().begin();
    if (!isa<ModOp>(modUser))
      return failure();

    // We can just bypass the current operand
    modUser->setOperand(0, modOp->getOperand(0));
    rewriter.eraseOp(modOp);
    return success();
  }
};

template <typename Op>
struct ForwardAddLike : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Type resType = op->getResult(0).getType();
    if (!isIntType(resType))
      return failure();

    // Check whether we can reduce the bitwidth of the operation
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value minLhs = getMinimalValue(lhs);
    Value minRhs = getMinimalValue(rhs);
    unsigned requiredWidth =
        std::max(minLhs.getType().getIntOrFloatBitWidth(),
                 minRhs.getType().getIntOrFloatBitWidth()) +
        1;
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (requiredWidth >= resWidth)
      return failure();

    // Create a new arithmetic operation as well as appropriate extension
    // operations to keep the IR valid
    rewriter.setInsertionPoint(op);
    Value newLhs = extendValue(minLhs, requiredWidth, rewriter)->getResult(0);
    Value newRhs = extendValue(minRhs, requiredWidth, rewriter)->getResult(0);
    auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
    Operation *extRes = extendValue(newOp->getResult(0), resWidth, rewriter);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(op, extRes->getResult(0));
    return success();
  }
};

template <typename Op>
struct BackwardAddLike : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Type resType = op->getResult(0).getType();
    if (!isIntType(resType))
      return failure();

    // Check whether we can reduce the bitwidth of the operation
    unsigned requiredWidth = getActualResultWidth(op->getResult(0));
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (requiredWidth >= resWidth)
      return failure();

    // Create a new arithmetic operation as well as appropriate extension
    // operations to keep the IR valid
    rewriter.setInsertionPoint(op);
    Value newLhs = modifyBitwidth(getMinimalValue(op->getOperand(0)),
                                  requiredWidth, rewriter);
    Value newRhs = modifyBitwidth(getMinimalValue(op->getOperand(1)),
                                  requiredWidth, rewriter);
    auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
    Operation *extRes = extendValue(newOp->getResult(0), resWidth, rewriter);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(op, extRes->getResult(0));
    return success();
  }
};

struct HandshakeOptimizeBitwidthsPass
    : public dynamatic::impl::HandshakeOptimizeBitwidthsBase<
          HandshakeOptimizeBitwidthsPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::ModuleOp modOp = getOperation();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
      bool fwChanged, bwChanged;
      SmallVector<Operation *> ops;
      do {
        // Forward pass
        fwChanged = false;
        RewritePatternSet fwPatterns{ctx};
        fwPatterns.add<ForwardAddLike<arith::AddIOp>>(ctx);
        ops.clear();
        for (Operation &op : funcOp.getOps())
          ops.push_back(&op);
        if (failed(applyOpPatternsAndFold(ops, std::move(fwPatterns), config,
                                          &fwChanged)))
          signalPassFailure();

        // Backward pass
        bwChanged = false;
        RewritePatternSet bwPatterns{ctx};
        bwPatterns.add<BackwardAddLike<arith::AddIOp>>(ctx);

        ops.clear();
        for (Operation &op : funcOp.getOps())
          ops.push_back(&op);
        if (failed(applyOpPatternsAndFold(ops, std::move(bwPatterns), config,
                                          &bwChanged)))
          signalPassFailure();

      } while (fwChanged || bwChanged);

      // Cleanup pass
      // RewritePatternSet modsPatterns{ctx};
      // modsPatterns.add<EraseUselessBitwidthModifier<arith::ExtSIOp>,
      //                  EraseUselessBitwidthModifier<arith::ExtUIOp>,
      //                  EraseUselessBitwidthModifier<arith::TruncIOp>,
      //                  CombineBitwidthModifiers<arith::ExtSIOp>,
      //                  CombineBitwidthModifiers<arith::ExtUIOp>,
      //                  CombineBitwidthModifiers<arith::TruncIOp>>(ctx);
      // if (failed(applyPatternsAndFoldGreedily(funcOp,
      // std::move(modsPatterns),
      //                                         config)))
      //   return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeOptimizeBitwidths() {
  return std::make_unique<HandshakeOptimizeBitwidthsPass>();
}
