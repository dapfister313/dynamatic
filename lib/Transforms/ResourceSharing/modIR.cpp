#include "dynamatic/Transforms/ResourceSharing/modIR.h"


std::map<int, dynamatic::sharing::controlStructure> modification_control_map;

void dynamatic::sharing::initialize_modification(std::map<int, dynamatic::sharing::controlStructure> control_map) {
  modification_control_map = control_map;
  return;
}

void dynamatic::sharing::revert_to_initial_state() {
  for(auto& item : modification_control_map) {
    item.second.current_position = item.second.control_merge;
  }
}

std::optional<unsigned> dynamatic::sharing::getLogicBB(Operation *op) {
  if (auto bb = op->getAttrOfType<mlir::IntegerAttr>(BB_ATTR))
    return bb.getUInt();
  return {};
}

Value dynamatic::sharing::generate_performance_step(OpBuilder* builder, mlir::Operation *op) {
  Value return_value;
  mlir::Value control_merge = modification_control_map[dynamatic::sharing::getLogicBB(op).value()].current_position;
  builder->setInsertionPointAfterValue(control_merge);
  //child operation of control merge
  mlir::Operation *child_op = control_merge.getUses().begin()->getOwner();
  //get control operands of newly created syncOp
  std::vector<Value> controlOperands = {control_merge};
  for(auto value : op->getOperands()) {
    controlOperands.push_back(value);
  }
  //create syncOp
  mlir::Operation *syncOp = builder->create<SyncOp>(control_merge.getLoc(), controlOperands);
  return_value = syncOp->getResult(0);
  //connect syncOp
  child_op->replaceUsesOfWith(control_merge, syncOp->getResult(0));
  int control_int = 0;
  for(auto value : op->getOperands()) {
    op->replaceUsesOfWith(value, syncOp->getResult(control_int+1));
    control_int++;
  }
  modification_control_map[dynamatic::sharing::getLogicBB(op).value()].current_position = syncOp->getResult(0);
  inheritBB(op, syncOp);
  return return_value;
}

std::vector<Value> dynamatic::sharing::generate_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items) {
  std::vector<Value> return_values;
  dynamatic::sharing::revert_to_initial_state();
  for(auto op : items) {
    return_values.push_back(dynamatic::sharing::generate_performance_step(builder, op));
  }
  return return_values;
}

void dynamatic::sharing::revert_performance_step(OpBuilder* builder, mlir::Operation *op) {
  //get specific syncOp
  mlir::Operation *syncOp = op->getOperand(0).getDefiningOp();
  //reconnect the previous state
  int control_int = 0;
  for(auto value : syncOp->getResults()) {
    value.replaceAllUsesWith(syncOp->getOperand(control_int));
    control_int++;
  }
  //delete syncOp
  syncOp->erase();
}

void dynamatic::sharing::destroy_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items) {
  for(auto op : items) {
    dynamatic::sharing::revert_performance_step(builder, op);
  }
}


//handshake::ForkOp
mlir::OpResult dynamatic::sharing::extend_fork(OpBuilder* builder, ForkOp OldFork) {
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

mlir::OpResult dynamatic::sharing::add_fork(OpBuilder* builder, mlir::OpResult connectionPoint) {
  return connectionPoint;
}

//handshake::SinkOp
void dynamatic::sharing::addSink(OpBuilder* builder, mlir::OpResult* connectionPoint) {
    builder->setInsertionPointAfter(connectionPoint->getOwner()->getOperand(0).getDefiningOp());
    auto newSinkOp = builder->create<SinkOp>(connectionPoint->getLoc(), *connectionPoint);
}

//handshake::ConstantOp
mlir::OpResult dynamatic::sharing::addConst(OpBuilder* builder, mlir::OpResult* connectionPoint, int value) {
    Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
    builder->setInsertionPointAfter(opSrc);
    IntegerAttr cond = builder->getBoolAttr(value);
    auto newConstOp = builder->create<handshake::ConstantOp>(connectionPoint->getLoc(), cond.getType(), cond, *connectionPoint);
    inheritBB(opSrc, newConstOp);
    return newConstOp->getResult(0);
}

//handshake::BranchOp
mlir::OpResult dynamatic::sharing::addBranch(OpBuilder* builder, mlir::OpResult* connectionPoint) {  //Value
    Operation *opSrc = connectionPoint->getOwner()->getOperand(0).getDefiningOp();
    auto newBranchOp = builder->create<handshake::BranchOp>(connectionPoint->getLoc(), *connectionPoint);
    inheritBB(opSrc, newBranchOp);
    return newBranchOp->getResult(0);
}

//handshake::MergeOp
mlir::OpResult dynamatic::sharing::addMerge(OpBuilder* builder, mlir::OpResult* connectionPoint) {
    return *connectionPoint;
}

void dynamatic::sharing::deleteAllBuffers(FuncOp funcOp) {
  for (auto op : llvm::make_early_inc_range(funcOp.getOps<BufferOp>())) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    //delete buffer
    op->erase();
  }
}

