//===- HandshakeOptimizeBitwidths.cpp - Optimize channel widths -*- C++ -*-===//
//
// TODO
//
// Note on shift operation handling (forward and backward): the logic of
// truncing a value only to extend it again immediately may seem unnecessary,
// but it in fact allows the rest of the rewrite patterns to understand that
// value fits on less bits than what the original value suggests. This is
// slightly convoluted but we are forced to do this like that since shift
// operations enforce that all their operands are of the same type. Ideally, we
// would have a Handshake version of shift operations that accept varrying
// bitwidths between its operands and result.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeOptimizeBitwidths.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>

using namespace mlir;
using namespace circt;
using namespace dynamatic;

static bool isIntType(Type type) { return isa<IntegerType>(type); }

static bool isBitwidthMod(Operation *op) {
  return isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op);
}

static bool isLogicOp(Operation *op) {
  return isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

// static void dematerializeBitwidthMods(handshake::FuncOp funcOp,
//                                       MLIRContext *ctx) {
//   OpBuilder builder(ctx);
//   for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
//     if (isBitwidthMod(&op)) {
//       op.getResult(0).replaceAllUsesWith(op.getOperand(0));
//       op.erase();
//     }
//   }
// }

// static void materializeBitwidthMods(handshake::FuncOp funcOp,
//                                     MLIRContext *ctx) {
//   OpBuilder builder(ctx);
//   // Materialize bitwidths modifiers for function arguments
//   Block &entryBlock = funcOp.front();
//   builder.setInsertionPointToStart(&entryBlock);
//   for (auto arg : entryBlock.getArguments())
//     addModsForValue(arg, funcOp->getLoc(), builder);

//   // Materialize bitwidths modifiers for all operations in the function body
//   for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
//     builder.setInsertionPointToStart(&entryBlock);
//     for (Value result : op.getResults())
//       addModsForValue(result, op.getLoc(), builder);
//   }
// }

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

static Value modVal(Value val, unsigned requiredWidth, bool logicExt,
                    PatternRewriter &rewriter) {
  Type type = val.getType();
  assert(isIntType(type) && "value must be integer type");

  unsigned width = type.getIntOrFloatBitWidth();
  Operation *newOp = nullptr;
  if (width < requiredWidth) {
    if (logicExt)
      newOp = rewriter.create<arith::ExtUIOp>(
          val.getLoc(), rewriter.getIntegerType(requiredWidth), val);
    else
      newOp = rewriter.create<arith::ExtSIOp>(
          val.getLoc(), rewriter.getIntegerType(requiredWidth), val);
  } else if (width > requiredWidth)
    newOp = rewriter.create<arith::TruncIOp>(
        val.getLoc(), rewriter.getIntegerType(requiredWidth), val);
  if (newOp) {
    inheritBBFromValue(val, newOp);
    return newOp->getResult(0);
  }
  return val;
}

template <typename Op>
static void modOp(Op op, Value lhs, Value rhs, unsigned width, bool logicExt,
                  PatternRewriter &rewriter) {
  Type resType = op->getResult(0).getType();
  assert(isa<IntegerType>(resType) && "result must have integer type");
  unsigned resWidth = resType.getIntOrFloatBitWidth();

  // Create a new operation as well as appropriate bitwidth modifications
  // operations to keep the IR valid
  rewriter.setInsertionPoint(op);
  Value newLhs = modVal(lhs, width, logicExt, rewriter);
  Value newRhs = modVal(rhs, width, logicExt, rewriter);
  auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
  Value newRes = modVal(newOp->getResult(0), resWidth, logicExt, rewriter);
  inheritBB(op, newOp);

  // Replace uses of the original operation's result with the result of the
  // optimized operation we just created
  rewriter.replaceOp(op, newRes);
}

//===----------------------------------------------------------------------===//
// Transfer functions
//===----------------------------------------------------------------------===//

static inline unsigned addWidth(unsigned lhs, unsigned rhs) {
  return std::max(lhs, rhs) + 1;
}

static inline unsigned mulWidth(unsigned lhs, unsigned rhs) {
  return lhs + rhs;
}

static inline unsigned divWidth(unsigned lhs, unsigned _) { return lhs + 1; }

static inline unsigned andWidth(unsigned lhs, unsigned rhs) {
  return std::min(lhs, rhs);
}

static inline unsigned orWidth(unsigned lhs, unsigned rhs) {
  return std::max(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

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

using FTransfer = std::function<unsigned(unsigned, unsigned)>;

template <typename Op>
struct SingleTypeArith : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  SingleTypeArith(bool forward, FTransfer fTransfer, MLIRContext *ctx)
      : OpRewritePattern<Op>(ctx), forward(forward),
        fTransfer(std::move(fTransfer)) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Type resType = op->getResult(0).getType();
    if (!isIntType(resType))
      return failure();

    // Check whether we can reduce the bitwidth of the operation
    Value minLhs = getMinimalValue(op->getOperand(0));
    Value minRhs = getMinimalValue(op->getOperand(1));
    unsigned requiredWidth;
    if (forward)
      requiredWidth = fTransfer(minLhs.getType().getIntOrFloatBitWidth(),
                                minRhs.getType().getIntOrFloatBitWidth());
    else
      requiredWidth = getActualResultWidth(op->getResult(0));
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (requiredWidth >= resWidth)
      return failure();

    // For bitwise logical operations, extension must also be logical
    bool logicExt = isLogicOp(op);
    modOp(op, minLhs, minRhs, requiredWidth, logicExt, rewriter);
    return success();
  }

private:
  bool forward;
  FTransfer fTransfer;
};

struct Select : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  Select(bool forward, MLIRContext *ctx)
      : OpRewritePattern<arith::SelectOp>(ctx), forward(forward) {}

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    Type resType = selectOp.getResult().getType();
    if (!isIntType(resType))
      return failure();

    // Check whether we can reduce the bitwidth of the operation
    Value minLhs = getMinimalValue(selectOp.getTrueValue());
    Value minRhs = getMinimalValue(selectOp.getFalseValue());
    unsigned requiredWidth;
    if (forward)
      requiredWidth = std::max(minLhs.getType().getIntOrFloatBitWidth(),
                               minRhs.getType().getIntOrFloatBitWidth());
    else
      requiredWidth = getActualResultWidth(selectOp.getResult());
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (requiredWidth >= resWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modifications
    // operations to keep the IR valid
    rewriter.setInsertionPoint(selectOp);
    Value newLhs = modVal(minLhs, requiredWidth, false, rewriter);
    Value newRhs = modVal(minRhs, requiredWidth, false, rewriter);
    auto newOp = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), newLhs, newRhs);
    Value newRes = modVal(newOp->getResult(0), resWidth, false, rewriter);
    inheritBB(selectOp, newOp);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(selectOp, newRes);
    return success();
  }

private:
  bool forward;
};

struct FWCmp : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    // Check whether we can reduce the bitwidth of the operation
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    Value minLhs = getMinimalValue(lhs);
    Value minRhs = getMinimalValue(rhs);
    unsigned requiredWidth = std::max(minLhs.getType().getIntOrFloatBitWidth(),
                                      minRhs.getType().getIntOrFloatBitWidth());
    unsigned actualWidth = lhs.getType().getIntOrFloatBitWidth();
    if (requiredWidth >= actualWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modifications
    // operations to keep the IR valid
    rewriter.setInsertionPoint(cmpOp);
    Value newLhs = modVal(minLhs, requiredWidth, false, rewriter);
    Value newRhs = modVal(minRhs, requiredWidth, false, rewriter);
    auto newOp = rewriter.create<arith::CmpIOp>(
        cmpOp.getLoc(), cmpOp.getPredicate(), newLhs, newRhs);
    inheritBB(cmpOp, newOp);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(cmpOp, newOp.getResult());
    return success();
  }
};

template <typename Op>
struct FWShift : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value toShift = op->getOperand(0);
    Value shiftBy = op->getOperand(1);
    Value minToShift = getMinimalValue(toShift);
    Value minShiftBy = getMinimalValue(shiftBy);
    bool isRightShift = isa<arith::ShRSIOp>(op) || isa<arith::ShRUIOp>(op);

    // Check whether we can reduce the bitwidth of the operation
    unsigned resWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    unsigned requiredWidth = resWidth;
    unsigned cstVal = 0;
    if (Operation *defOp = minShiftBy.getDefiningOp())
      if (auto cstOp = dyn_cast<handshake::ConstantOp>(defOp)) {
        cstVal = (unsigned)cast<IntegerAttr>(cstOp.getValue()).getInt();
        unsigned baseWidth = minToShift.getType().getIntOrFloatBitWidth();
        if (isRightShift)
          requiredWidth = baseWidth;
        else
          requiredWidth = baseWidth + cstVal;
      }
    if (requiredWidth >= resWidth)
      return failure();

    llvm::outs() << "Required width is " << requiredWidth << "\n";
    // For logical shifts, extension must also be logical
    bool logicExt = isa<arith::ShLIOp>(op) || isa<arith::ShRUIOp>(op);
    // modOp(op, minToShift, minShiftBy, requiredWidth, logicExt, rewriter);

    // Create a new operation as well as appropriate bitwidth modifications
    // operations to keep the IR valid
    rewriter.setInsertionPoint(op);
    Value newLhs = modVal(minToShift, requiredWidth, logicExt, rewriter);
    Value newRhs = modVal(minShiftBy, requiredWidth, logicExt, rewriter);
    auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
    Value newRes = newOp->getResult(0);
    if (isRightShift)
      // In the case of a right shift, we can first truncate the result of the
      // newly inserted shift operation to discard high-order bits that we know
      // are 0s, then extend the result back to satisfy the users of the
      // original operation's result
      newRes = modVal(newRes, requiredWidth - cstVal, logicExt, rewriter);
    Value modRes = modVal(newRes, resWidth, logicExt, rewriter);
    inheritBB(op, newOp);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(op, modRes);
    return success();
  }
};

template <typename Op>
struct BWShift : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value toShift = op->getOperand(0);
    Value shiftBy = op->getOperand(1);
    Value minToShift = getMinimalValue(toShift);
    Value minShiftBy = getMinimalValue(shiftBy);
    bool isRightShift = isa<arith::ShRSIOp>(op) || isa<arith::ShRUIOp>(op);

    // Check whether we can reduce the bitwidth of the operation
    unsigned resWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    unsigned requiredWidth = resWidth;
    unsigned cstVal = 0;
    if (Operation *defOp = minShiftBy.getDefiningOp())
      if (auto cstOp = dyn_cast<handshake::ConstantOp>(defOp)) {
        cstVal = (unsigned)cast<IntegerAttr>(cstOp.getValue()).getInt();
        unsigned baseWidth = getActualResultWidth(op->getResult(0));
        if (isRightShift)
          requiredWidth = baseWidth + cstVal;
        else
          requiredWidth = baseWidth;
      }
    if (requiredWidth >= resWidth)
      return failure();

    // Compute the number of bits actually required for the shifted integer
    if (!isRightShift) {
    }

    // For logical shifts, extension must also be logical
    bool logicExt = isa<arith::ShLIOp>(op) || isa<arith::ShRUIOp>(op);

    Value modToShift = minToShift;
    if (!isRightShift) {
      // In the case of a right shift, we first truncate the shifted integer to
      // discard high-order bits that were discarded in the result, then extend
      // back to satisfy the users of the original integer
      unsigned requiredToShiftWidth =
          requiredWidth - std::min(cstVal, requiredWidth);
      modToShift = modVal(minToShift, requiredToShiftWidth, logicExt, rewriter);
    }
    modOp(op, modToShift, minShiftBy, requiredWidth, logicExt, rewriter);
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
        fwPatterns.add<SingleTypeArith<arith::AddIOp>,
                       SingleTypeArith<arith::SubIOp>>(true, addWidth, ctx);
        fwPatterns.add<SingleTypeArith<arith::MulIOp>>(true, mulWidth, ctx);
        fwPatterns.add<
            SingleTypeArith<arith::DivUIOp>, SingleTypeArith<arith::DivSIOp>,
            SingleTypeArith<arith::RemUIOp>, SingleTypeArith<arith::RemSIOp>>(
            true, divWidth, ctx);
        fwPatterns.add<SingleTypeArith<arith::AndIOp>>(true, andWidth, ctx);
        fwPatterns
            .add<SingleTypeArith<arith::OrIOp>, SingleTypeArith<arith::XOrIOp>>(
                true, orWidth, ctx);
        fwPatterns.add<FWShift<arith::ShLIOp>, FWShift<arith::ShRSIOp>,
                       FWShift<arith::ShRUIOp>, FWCmp>(ctx);
        fwPatterns.add<Select>(true, ctx);

        ops.clear();
        for (Operation &op : funcOp.getOps())
          ops.push_back(&op);
        if (failed(applyOpPatternsAndFold(ops, std::move(fwPatterns), config,
                                          &fwChanged)))
          signalPassFailure();

        // Backward pass
        bwChanged = false;
        RewritePatternSet bwPatterns{ctx};
        bwPatterns
            .add<SingleTypeArith<arith::AddIOp>, SingleTypeArith<arith::SubIOp>,
                 SingleTypeArith<arith::MulIOp>, SingleTypeArith<arith::AndIOp>,
                 SingleTypeArith<arith::OrIOp>, SingleTypeArith<arith::XOrIOp>>(
                false, addWidth, ctx);
        bwPatterns.add<BWShift<arith::ShLIOp>, BWShift<arith::ShRSIOp>,
                       BWShift<arith::ShRUIOp>>(ctx);
        bwPatterns.add<Select>(false, ctx),

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
