//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "dynamatic/Transforms/ForwardUpdate.h"
#include "dynamatic/Transforms/BackwardUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/IndentedOstream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

using namespace mlir::detail;


static LogicalResult rewriteBitsWidths(handshake::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  SmallVector<Operation *> vecOp;

  using forward_func  = std::function<unsigned (mlir::Operation::operand_range vecOperands)>;
  using backward_func = std::function<unsigned (mlir::Operation::result_range vecResults)>;
  
  SmallVector<Operation *> containerOps;
  

  bool changed = true;
  int savedBits = 0;
  while (changed) {
    // init 
    changed = false;
    containerOps.clear();

    for (auto &op : funcOp.getOps())
      containerOps.push_back(&op);
    
    // Forward process
    DenseMap<StringRef, forward_func> forMapOpNameWidth;
    forward::constructFuncMap(forMapOpNameWidth);
    for (auto opPointer=containerOps.begin(); opPointer!=containerOps.end(); ++opPointer){
      
      auto op = *opPointer;

      if (isa<handshake::ConstantOp>(*op))
        continue;

      if (update::propType(op))
        continue;

      if (isa<mlir::arith::ExtSIOp>(op) || isa<mlir::arith::ExtUIOp>(op)) {
        update::replaceWithSuccessor(op);
        op->erase();
        continue;
      }

      const auto opName = op->getName().getStringRef();
      unsigned int newWidth = 0, resInd=0;
      if (forMapOpNameWidth.find(opName) != forMapOpNameWidth.end())
        newWidth = forMapOpNameWidth[opName](op->getOperands());


      if (isa<handshake::ControlMergeOp>(op))
        resInd = 1; 
      // if the new type can be optimized, update the type
      if (newWidth>0)
        if(Type newOpResultType = getNewType(op->getResult(resInd), newWidth, true);  
            newWidth < op->getResult(resInd).getType().getIntOrFloatBitWidth() ){  
          changed |= true;
          savedBits += op->getResult(resInd).getType().getIntOrFloatBitWidth()-newWidth;
          op->getResult(resInd).setType(newOpResultType);

        }
    }

    // for (auto &op : funcOp.getOps())
    //   llvm::errs() << op <<"\n";

    // Backward Process
    DenseMap<StringRef, backward_func> backMapOpNameWidth;
    backward::constructFuncMap(backMapOpNameWidth);

    for (auto opPointer=containerOps.rbegin(); opPointer!=containerOps.rend(); ++opPointer) {
      auto op = *opPointer;

      if (isa<handshake::ConstantOp>(*op))
        continue;
      
     if (isa<mlir::arith::TruncIOp>(*op)) {
        // op->getOperand(0).setType(op->getResult(0).getType());
        update::replaceWithSuccessor(op, op->getResult(0).getType());
        op->erase();
        continue;
      }

      const auto opName = op->getName().getStringRef();

      unsigned int newWidth = 0;
      // get the new bit width of the result operator
      if (backMapOpNameWidth.find(opName) != backMapOpNameWidth.end())
        newWidth = backMapOpNameWidth[opName](op->getResults());

      if (newWidth>0)
        // if the new type can be optimized, update the type
        if(Type newOpResultType = getNewType(op->getOperand(0), newWidth, true);  
            newWidth < op->getOperand(0).getType().getIntOrFloatBitWidth()){
          changed |= true;
          // llvm::errs() << "backward op " << *op << "\n";
          for (unsigned i=0;i<op->getNumOperands();++i)
            if (newWidth < op->getOperand(i).getType().getIntOrFloatBitWidth()) {
              savedBits += op->getOperand(i).getType().getIntOrFloatBitWidth()-newWidth;
              op->getOperand(i).setType(newOpResultType);
            }
        }
      
    }

    llvm::errs() << "-----------------end one round-----------------\n";

  }
  // Store new inserted truncation or extension operation during validation
  SmallVector<Operation *> OpTruncExt;
  for (auto &op : funcOp.getOps()) {
      // llvm::errs() <<"before validation : " << op <<"\n";
      update::validateOp(&op, ctx, OpTruncExt);
      // llvm::errs() <<"after validation : " << op <<"\n\n";
  }

  // Validate the new inserted operation
  for (auto op : OpTruncExt)
    update::validateOp(op, ctx, OpTruncExt);

  llvm::errs() << "Total saved bits " << savedBits << "\n";
  
  return success();
}

struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if(failed(rewriteBitsWidths(funcOp, ctx)))
        return signalPassFailure();


  };

};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}