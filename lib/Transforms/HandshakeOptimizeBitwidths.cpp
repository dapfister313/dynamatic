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

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static bool isIntType(Type type) { return isa<IntegerType>(type); }

static bool isLogicOp(Operation *op) {
  return isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
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
  assert(isIntType(resType) && "result must have integer type");
  unsigned resWidth = resType.getIntOrFloatBitWidth();

  // Create a new operation as well as appropriate bitwidth modification
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
// Transfer functions for arith operations
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

namespace {

//===----------------------------------------------------------------------===//
// Handshake-dialect patterns
//===----------------------------------------------------------------------===//

template <typename Op>
class OptDataConfig {
public:
  OptDataConfig() = default;

  virtual SmallVector<Value> getDataOperands(Op op) const {
    return op->getOperands();
  }

  virtual SmallVector<Value> getDataResults(Op op) const {
    return op->getResults();
  }

  virtual void getNewOperands(Op op, unsigned width,
                              SmallVector<Value> &minDataOperands,
                              PatternRewriter &rewriter,
                              SmallVector<Value> &newOperands) const {
    llvm::transform(
        minDataOperands, std::back_inserter(newOperands),
        [&](Value val) { return modVal(val, width, false, rewriter); });
  }

  virtual void getResultTypes(Op op, Type type,
                              SmallVector<Type> &newResTypes) const {
    for (size_t i = 0, numResults = op->getNumResults(); i < numResults; ++i)
      newResTypes.push_back(type);
  }

  virtual Op createOp(Op op, SmallVector<Type> &newResTypes,
                      SmallVector<Value> &newOperands,
                      PatternRewriter &rewriter) const {
    return rewriter.create<Op>(op.getLoc(), newResTypes, newOperands);
  }

  virtual void modResults(Op newOp, unsigned width, PatternRewriter &rewriter,
                          SmallVector<Value> &newResults) const {
    llvm::transform(
        newOp->getResults(), std::back_inserter(newResults),
        [&](OpResult res) { return modVal(res, width, false, rewriter); });
  }

  virtual ~OptDataConfig() = default;
};

class CMergeDataConfig : public OptDataConfig<handshake::ControlMergeOp> {
public:
  SmallVector<Value>
  getDataResults(handshake::ControlMergeOp op) const override {
    return SmallVector<Value>{op.getResult()};
  }

  void getResultTypes(handshake::ControlMergeOp op, Type type,
                      SmallVector<Type> &newResTypes) const override {
    for (size_t i = 0, numResults = op->getNumResults() - 1; i < numResults;
         ++i)
      newResTypes.push_back(type);
    newResTypes.push_back(op.getIndex().getType());
  }

  void modResults(handshake::ControlMergeOp newOp, unsigned width,
                  PatternRewriter &rewriter,
                  SmallVector<Value> &newResults) const override {
    newResults.push_back(modVal(newOp.getResult(), width, false, rewriter));
    newResults.push_back(newOp.getIndex());
  }
};

class MuxDataConfig : public OptDataConfig<handshake::MuxOp> {
public:
  SmallVector<Value> getDataOperands(handshake::MuxOp op) const override {
    return op.getDataOperands();
  }

  void getNewOperands(handshake::MuxOp op, unsigned width,
                      SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) const override {
    newOperands.push_back(op.getSelectOperand());
    llvm::transform(
        minDataOperands, std::back_inserter(newOperands),
        [&](Value val) { return modVal(val, width, false, rewriter); });
  }
};

class CBranchDataConfig : public OptDataConfig<handshake::ConditionalBranchOp> {
public:
  SmallVector<Value>
  getDataOperands(handshake::ConditionalBranchOp op) const override {
    return SmallVector<Value>{op.getDataOperand()};
  }

  void getNewOperands(handshake::ConditionalBranchOp op, unsigned width,
                      SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) const override {
    newOperands.push_back(op.getConditionOperand());
    newOperands.push_back(modVal(minDataOperands[0], width, false, rewriter));
  }
};

class BufferDataConfig : public OptDataConfig<handshake::BufferOp> {
public:
  handshake::BufferOp createOp(handshake::BufferOp bufOp,
                               SmallVector<Type> &newResTypes,
                               SmallVector<Value> &newOperands,
                               PatternRewriter &rewriter) const override {
    return rewriter.create<handshake::BufferOp>(bufOp.getLoc(), newOperands[0],
                                                bufOp.getNumSlots(),
                                                bufOp.getBufferType());
  }
};

template <typename Op, typename Cfg>
struct HandshakeOptData : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  HandshakeOptData(bool forward, MLIRContext *ctx)
      : OpRewritePattern<Op>(ctx), forward(forward), cfg(Cfg()) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> dataOperands = cfg.getDataOperands(op);
    SmallVector<Value> dataResults = cfg.getDataResults(op);
    assert(!dataOperands.empty() && "op must have at least one data operand");
    assert(!dataResults.empty() && "op must have at least one data result");

    Type dataType = dataResults[0].getType();
    if (!isIntType(dataType))
      return failure();

    // Get the operation's data operands actual widths
    SmallVector<Value> minDataOperands;
    llvm::transform(dataOperands, std::back_inserter(minDataOperands),
                    [&](Value val) { return getMinimalValue(val); });

    // Check whether we can reduce the bitwidth of the operation
    unsigned requiredWidth = 0;
    if (forward) {
      for (Value opr : minDataOperands)
        requiredWidth =
            std::max(requiredWidth, opr.getType().getIntOrFloatBitWidth());
    } else {
      for (Value res : dataResults)
        requiredWidth = std::max(requiredWidth, getActualResultWidth(res));
    }
    unsigned dataWidth = dataType.getIntOrFloatBitWidth();
    if (requiredWidth >= dataWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    rewriter.setInsertionPoint(op);
    SmallVector<Value> newOperands, newResults;
    SmallVector<Type> newResTypes;
    cfg.getNewOperands(op, requiredWidth, minDataOperands, rewriter,
                       newOperands);
    cfg.getResultTypes(op, rewriter.getIntegerType(requiredWidth), newResTypes);
    Op newOp = cfg.createOp(op, newResTypes, newOperands, rewriter);
    inheritBB(op, newOp);
    cfg.modResults(newOp, dataWidth, rewriter, newResults);

    // Replace uses of the original operation's results with the results of the
    // optimized operation we just created
    rewriter.replaceOp(op, newResults);
    return success();
  }

private:
  bool forward;
  Cfg cfg;
};

/// Template specialization of data optimization rewrite pattern for Handshake
/// operations that do not require a specific configuration.
template <typename Op>
using HandshakeOptDataNoCfg = HandshakeOptData<Op, OptDataConfig<Op>>;

//===----------------------------------------------------------------------===//
// arith-dialect patterns
//===----------------------------------------------------------------------===//

using FTransfer = std::function<unsigned(unsigned, unsigned)>;

template <typename Op>
struct ArithSingleType : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  ArithSingleType(bool forward, FTransfer fTransfer, MLIRContext *ctx)
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

    // Create a new operation as well as appropriate bitwidth modification
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

    // Create a new operation as well as appropriate bitwidth modification
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

    // Create a new operation as well as appropriate bitwidth modification
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

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
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
        fillForwardPatterns(fwPatterns);
        fillFuncOps(funcOp, ops);
        if (failed(applyOpPatternsAndFold(ops, std::move(fwPatterns), config,
                                          &fwChanged)))
          signalPassFailure();

        // Backward pass
        bwChanged = false;
        RewritePatternSet bwPatterns{ctx};
        fillBackwardPatterns(bwPatterns);
        fillFuncOps(funcOp, ops);
        if (failed(applyOpPatternsAndFold(ops, std::move(bwPatterns), config,
                                          &bwChanged)))
          signalPassFailure();

      } while (fwChanged || bwChanged);
    }
  }

private:
  void fillHandshakeDataPatterns(RewritePatternSet &patterns, bool forward) {
    MLIRContext *ctx = patterns.getContext();

    patterns.add<HandshakeOptDataNoCfg<handshake::ForkOp>,
                 HandshakeOptDataNoCfg<handshake::LazyForkOp>,
                 HandshakeOptDataNoCfg<handshake::MergeOp>,
                 HandshakeOptDataNoCfg<handshake::BranchOp>>(forward, ctx);
    patterns.add<HandshakeOptData<handshake::ControlMergeOp, CMergeDataConfig>>(
        forward, ctx);
    patterns.add<HandshakeOptData<handshake::MuxOp, MuxDataConfig>>(forward,
                                                                    ctx);
    patterns.add<
        HandshakeOptData<handshake::ConditionalBranchOp, CBranchDataConfig>>(
        forward, ctx);
    patterns.add<HandshakeOptData<handshake::BufferOp, BufferDataConfig>>(
        forward, ctx);
  }

  void fillForwardPatterns(RewritePatternSet &fwPatterns) {
    MLIRContext *ctx = fwPatterns.getContext();

    // Handshake operations
    fillHandshakeDataPatterns(fwPatterns, true);

    // arith operations
    fwPatterns
        .add<ArithSingleType<arith::AddIOp>, ArithSingleType<arith::SubIOp>>(
            true, addWidth, ctx);
    fwPatterns.add<ArithSingleType<arith::MulIOp>>(true, mulWidth, ctx);
    fwPatterns
        .add<ArithSingleType<arith::DivUIOp>, ArithSingleType<arith::DivSIOp>,
             ArithSingleType<arith::RemUIOp>, ArithSingleType<arith::RemSIOp>>(
            true, divWidth, ctx);
    fwPatterns.add<ArithSingleType<arith::AndIOp>>(true, andWidth, ctx);
    fwPatterns
        .add<ArithSingleType<arith::OrIOp>, ArithSingleType<arith::XOrIOp>>(
            true, orWidth, ctx);
    fwPatterns.add<FWShift<arith::ShLIOp>, FWShift<arith::ShRSIOp>,
                   FWShift<arith::ShRUIOp>, FWCmp>(ctx);
    fwPatterns.add<Select>(true, ctx);
  }

  void fillBackwardPatterns(RewritePatternSet &bwPatterns) {
    MLIRContext *ctx = bwPatterns.getContext();

    // Handshake operations
    fillHandshakeDataPatterns(bwPatterns, false);

    // arith operations
    bwPatterns
        .add<ArithSingleType<arith::AddIOp>, ArithSingleType<arith::SubIOp>,
             ArithSingleType<arith::MulIOp>, ArithSingleType<arith::AndIOp>,
             ArithSingleType<arith::OrIOp>, ArithSingleType<arith::XOrIOp>>(
            false, addWidth, ctx);
    bwPatterns.add<BWShift<arith::ShLIOp>, BWShift<arith::ShRSIOp>,
                   BWShift<arith::ShRUIOp>>(ctx);
    bwPatterns.add<Select>(false, ctx);
  }

  void fillFuncOps(handshake::FuncOp funcOp, SmallVector<Operation *> &ops) {
    ops.clear();
    llvm::transform(funcOp.getOps(), std::back_inserter(ops),
                    [&](Operation &op) { return &op; });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeOptimizeBitwidths() {
  return std::make_unique<HandshakeOptimizeBitwidthsPass>();
}
