#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/IndentedOstream.h"


IntegerType getNewType(Value opType, 
                       unsigned bitswidth, 
                       bool signless) 
{                 
  IntegerType::SignednessSemantics ifSign = 
  IntegerType::SignednessSemantics::Signless;
  if (!signless)
    if (auto validType = opType.getType() ; isa<IntegerType>(validType))
      ifSign = dyn_cast<IntegerType>(validType).getSignedness();

  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

IntegerType getNewType(Value opType, 
                       unsigned bitswidth,  
                       IntegerType::SignednessSemantics ifSign) 
{
  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

// specify which value to extend
std::optional<Operation *> insertWidthMatchOp (Operation *newOp, 
                                               int opInd, 
                                               Type newType, 
                                               MLIRContext *ctx)
{
  OpBuilder builder(ctx);
  Value opVal = newOp->getOperand(opInd);
  
  unsigned int opWidth;
  if (isa<IndexType>(opVal.getType()))
    opWidth = 64;
  else
    opWidth = opVal.getType().getIntOrFloatBitWidth();
  
  if (isa<IntegerType>(opVal.getType()) || isa<IndexType>(opVal.getType())){
    // insert Truncation operation to match the opresult width
    if (opWidth > newType.getIntOrFloatBitWidth()){
      builder.setInsertionPoint(newOp);
      auto truncOp = builder.create<mlir::arith::TruncIOp>(newOp->getLoc(), 
                                                        newType,
                                                        opVal); 
      if (!isa<IndexType>(opVal.getType()))
        newOp->setOperand(opInd, truncOp.getResult());
        
      return truncOp;
    } 

    // insert Extension operation to match the opresult width
    if (opWidth < newType.getIntOrFloatBitWidth()){
      builder.setInsertionPoint(newOp);
      auto extOp = builder.create<mlir::arith::ExtSIOp>(newOp->getLoc(),
                                          newType,
                                          opVal); 
      
      if (!isa<IndexType>(opVal.getType())) 
        newOp->setOperand(opInd, extOp.getResult());
      
      return extOp;
    }
  }
  return {};    

}

namespace update {

void constructFuncMap(DenseMap<StringRef, 
                     std::function<std::vector<std::vector<unsigned int>> 
                                  (Operation::operand_range vecOperands, 
                                  Operation::result_range vecResults)>> 
                     &mapOpNameWidth){
  
  mapOpNameWidth[StringRef("arith.addi")] = 
    [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                                         vecOperands[1].getType().getIntOrFloatBitWidth());
                                
      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                           maxOpWidth+1);

      width = std::min(cpp_max_width, width);
      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

  mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];
   
  mapOpNameWidth[StringRef("arith.muli")] = 
    [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = vecOperands[0].getType().getIntOrFloatBitWidth() + 
                                vecOperands[1].getType().getIntOrFloatBitWidth();
                                
      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                           maxOpWidth);
      
      width = std::min(cpp_max_width, width);

      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("arith.ceildivsi")] = 
    [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = vecOperands[0].getType().getIntOrFloatBitWidth();
                                
      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                           maxOpWidth+1);
      
      width = std::min(cpp_max_width, width);

      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("arith.ceildivusi")] = mapOpNameWidth[StringRef("arith.ceildivsi")];

    mapOpNameWidth[StringRef("arith.cmpi")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                                         vecOperands[1].getType().getIntOrFloatBitWidth());
      
      unsigned int width = std::min(cpp_max_width, width);

      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({unsigned(1)}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("handshake.mux")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

      std::vector<std::vector<unsigned>> widths; 
      unsigned maxOpWidth = 2;

      unsigned ind = 0; // record number of operators

      for (auto oprand : vecOperands) {
        ind++;
        if (!isa<NoneType>(oprand.getType()))
          if (!isa<IndexType>(oprand.getType()) && 
              oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
            maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
      }
      unsigned indexWidth;
      if (ind>0)
        indexWidth = log2(ind-1)+2;
      else
        indexWidth = 2;
      widths.push_back({indexWidth}); // the bit width for the mux index result;

      if (isa<NoneType>(vecOperands[0].getType())) {
        widths.push_back({});
        return widths;
      }

      unsigned int width = std::min(cpp_max_width, maxOpWidth);
      // 1st operand is the index; rest of (ind -1) operands set to width
      std::vector<unsigned> opwidths(ind-1, width); 

      widths[0].insert(widths[0].end(), opwidths.begin(), opwidths.end()); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

     mapOpNameWidth[StringRef("handshake.merge")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

      std::vector<std::vector<unsigned>> widths; 
      unsigned maxOpWidth = 2;

      unsigned ind = 0; // record number of operators

      for (auto oprand : vecOperands) {
        ind++;
        if (!isa<NoneType>(vecOperands[0].getType()))
          if (!isa<IndexType>(oprand.getType()) && 
              oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
            maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
      }

      if (isa<NoneType>(vecOperands[0].getType())) {
        widths.push_back({});
        widths.push_back({});
        return widths;
      }

      unsigned int width = std::min(cpp_max_width, maxOpWidth);
      std::vector<unsigned> opwidths(ind, width);

      widths.push_back(opwidths); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("handshake.d_return")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

          std::vector<std::vector<unsigned>> widths; 
          widths.push_back({address_width});
          widths.push_back({address_width});
          return widths;
    };

    mapOpNameWidth[StringRef("handshake.d_load")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

          std::vector<std::vector<unsigned>> widths; 
          widths.push_back({cpp_max_width, address_width});
          widths.push_back({cpp_max_width, address_width});
          return widths;
    };

    mapOpNameWidth[StringRef("handshake.d_store")] = mapOpNameWidth[StringRef("handshake.d_load")];

};

  void setValidateType(Operation *Op,
                       bool &pass,
                       bool &match,
                       bool &revert) {

    pass   = false;
    match  = false;
    revert = false;
 
    if (isa<handshake::BranchOp>(*Op) ||
        isa<handshake::ConditionalBranchOp>(*Op))
        pass = true;

    if (isa<mlir::arith::AddIOp>(*Op)  ||
        isa<mlir::arith::SubIOp>(*Op)  ||
        isa<mlir::arith::MulIOp>(*Op)  ||
        isa<mlir::arith::DivSIOp>(*Op) ||
        isa<mlir::arith::DivUIOp>(*Op) ||
        isa<mlir::arith::CmpIOp>(*Op)  ||
        isa<handshake::MuxOp>(*Op)     ||
        isa<handshake::MergeOp>(*Op)   ||
        isa<handshake::DynamaticLoadOp>(*Op) ||
        isa<handshake::DynamaticStoreOp>(*Op) ||
        isa<handshake::DynamaticReturnOp>(*Op) )
      match = true;  

    if (isa<mlir::arith::TruncIOp>(*Op) ||
        isa<mlir::arith::ExtSIOp>(*Op) ||
        isa<mlir::arith::ExtUIOp>(*Op))
      revert = true;  
  }

  bool propType(Operation *op) {

    if (isa<handshake::ConditionalBranchOp>(*op)) {
      for (auto resOp : op->getResults())
        resOp.setType(op->getOperand(1).getType());
      return true;
    }

    if (isa<handshake::BranchOp>(*op)) {
      op->getResult(0).setType(op->getOperand(0).getType());
      return true;
    }
    return false;
  }

  void revertTruncOrExt(Operation *Op, MLIRContext *ctx) {
    OpBuilder builder(ctx);
    // if width(res) == width(opr) : delte the operand;
    if (Op->getResult(0).getType().getIntOrFloatBitWidth() ==
        Op->getOperand(0).getType().getIntOrFloatBitWidth()) {
      for(auto user : Op->getResult(0).getUsers()) 
        user->replaceUsesOfWith(Op->getResult(0), Op->getOperand(0));
      Op->erase();
    }

    // if for extension operation width(res) < width(opr),
    // change it to truncation operation
    if (isa<mlir::arith::ExtSIOp>(*Op) || isa<mlir::arith::ExtUIOp>(*Op))
      if (Op->getResult(0).getType().getIntOrFloatBitWidth() <
          Op->getOperand(0).getType().getIntOrFloatBitWidth()) {

        builder.setInsertionPoint(Op);
        Type newType = getNewType(Op->getResult(0), 
                                  Op->getResult(0).getType().getIntOrFloatBitWidth(), 
                                  false);
        auto truncOp = builder.create
                      <mlir::arith::TruncIOp>(Op->getLoc(), 
                                              newType,
                                              Op->getOperand(0));
        Op->getResult(0).replaceAllUsesWith(truncOp.getResult());
        Op->erase();
      }

    // if for truncation operation width(res) > width(opr),
    // change it to extension operation
    if (isa<mlir::arith::TruncIOp>(*Op))
      if (Op->getResult(0).getType().getIntOrFloatBitWidth() >
          Op->getOperand(0).getType().getIntOrFloatBitWidth()) {
            
        builder.setInsertionPoint(Op);
        Type newType = getNewType(Op->getResult(0), 
                                  Op->getResult(0).getType().getIntOrFloatBitWidth(), 
                                  false);
        auto truncOp = builder.create
                      <mlir::arith::ExtSIOp>(Op->getLoc(), 
                                              newType,
                                              Op->getOperand(0));
        Op->getResult(0).replaceAllUsesWith(truncOp.getResult());
        Op->erase();
      }
  }

  void matchOpResWidth (Operation *Op, MLIRContext *ctx) {

    DenseMap<mlir::StringRef,
               std::function<std::vector<std::vector<unsigned int>> 
                  (Operation::operand_range vecOperands, 
                   Operation::result_range vecResults)>> mapOpNameWidth;

    constructFuncMap(mapOpNameWidth);

    std::vector<std::vector<unsigned int> > OprsWidth = 
                                   mapOpNameWidth[Op->getName().getStringRef()]
                                   (Op->getOperands(), Op->getResults());
    // make operator matched the width

    for (unsigned int i = 0; i < OprsWidth[0].size(); ++i) {
      llvm::errs() <<  i << "\n";
      if (auto Operand = Op->getOperand(i); 
          Operand.getType().getIntOrFloatBitWidth() != OprsWidth[0][i])
        auto insertOp = insertWidthMatchOp(Op, 
                                           i, 
                                           getNewType(Operand, OprsWidth[0][i], false), 
                                           ctx);
    }
    // make result matched the width
    for (unsigned int i = 0; i < OprsWidth[1].size(); ++i) {
      llvm::errs() <<  Op->getResult(i) << "adapted to " << OprsWidth[1][i] << "\n";
      if (auto OpRes = Op->getResult(i); 
          OpRes.getType().getIntOrFloatBitWidth() != OprsWidth[1][i]) {
        Type newType = getNewType(OpRes, OprsWidth[1][i], false);
        Op->getResult(i).setType(newType) ;
      }
    }
  }

  void validateOp(Operation *Op, MLIRContext *ctx) {
    // the operations can be divided to three types to make it validated
    // passType: branch, conditionalbranch
    // c <= op(a,b): addi, subi, mux, etc. where both a,b,c needed to be verified
    // need to be reverted or deleted : truncIOp, extIOp
    bool pass, match, revert;
    setValidateType(Op, pass, match, revert);

    if (pass)
      bool res = propType(Op);

    if (match)
      matchOpResWidth(Op, ctx);

    if (revert) 
      revertTruncOrExt(Op, ctx);
      
  }
}