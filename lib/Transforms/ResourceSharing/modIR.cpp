#include "dynamatic/Transforms/ResourceSharing/modIR.h"

//handshake::ForkOp
mlir::OpResult extend_fork(OpBuilder* builder, ForkOp OldFork) {
      Operation *opSrc = OldFork.getOperand().getDefiningOp();
      Value opSrcIn = OldFork.getOperand();
      std::vector<Operation *> opsToProcess;
      for (auto &u : OldFork.getResults().getUses())
        opsToProcess.push_back(u.getOwner());
    
      // Insert fork after op
      builder->setInsertionPointAfter(opSrc);
      auto forkSize = opsToProcess.size();
      
      auto newForkOp = builder->create<ForkOp>(opSrcIn.getLoc(), opSrcIn, forkSize + 1);
      inheritBB(opSrc, newForkOp);
      for (int i = 0, e = forkSize; i < e; ++i)
        opsToProcess[i]->replaceUsesOfWith(OldFork->getResult(i), newForkOp->getResult(i));
      OldFork.erase();
      return newForkOp->getResult(forkSize);
}

//handshake::SinkOp
void addSink(OpBuilder* builder, mlir::OpResult* connectionPoint) {
    builder->setInsertionPointAfter(connectionPoint->getOwner()->getOperand(0).getDefiningOp());
    auto newSinkOp = builder->create<SinkOp>(connectionPoint->getLoc(), *connectionPoint);
}

//handshake::ConstantOp
mlir::OpResult addConst(OpBuilder* builder, mlir::OpResult* connectionPoint, int value) {
    Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
    builder->setInsertionPointAfter(opSrc);
    IntegerAttr cond = builder->getBoolAttr(value);
    auto newConstOp = builder->create<handshake::ConstantOp>(connectionPoint->getLoc(), cond.getType(), cond, *connectionPoint);
    inheritBB(opSrc, newConstOp);
    return newConstOp->getResult(0);
}

//handshake::BranchOp
mlir::OpResult addBranch(OpBuilder* builder, mlir::OpResult* connectionPoint) {  //Value
    Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
    auto newBranchOp = builder->create<handshake::BranchOp>(connectionPoint->getLoc(), *connectionPoint);
    inheritBB(opSrc, newBranchOp);
    return newBranchOp->getResult(0);
}

//handshake::MergeOp
mlir::OpResult addMerge(OpBuilder* builder, mlir::OpResult* connectionPoint) {
    return *connectionPoint;
}
