//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

static LogicalResult insertBuffers(handshake::FuncOp funcOp, 
                                   MLIRContext *ctx,
                                   bool firstMG,
                                   std::string stdLevelInfo) {

  std::vector<Operation *> visitedOpList;
  std::vector<unit *> unitList;

  // DFS build the dataflowCircuit from the handshake level
  for (auto &op : funcOp.getOps())
    if (isEntryOp(&op, visitedOpList))
      dfsHandshakeGraph(&op, unitList, visitedOpList);

  // create CFDFC circuits
  std::vector<dataFlowCircuit *> dataFlowCircuitList;
  // speficy by a flag, whether read the bb file from std level
  if (stdLevelInfo!=""){
    std::map<archBB*, int> archs;
    std::map<int, int> bbs;

    readSimulateFile(stdLevelInfo, archs, bbs);
    int execNum = buffer::extractCFDFCircuit(archs, bbs);
    while (execNum > 0) {
      dataFlowCircuitList.push_back(createCFDFCircuit(unitList, archs, bbs));
      if (firstMG)
        break;
      execNum = buffer::extractCFDFCircuit(archs, bbs);
    }
  }
  
  return success();
}


namespace {
/// Simple driver for prepare for legacy pass.
struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  PlaceBuffersPass(bool firstMG, std::string stdLevelInfo) {
    this->firstMG = firstMG;
    this->stdLevelInfo = stdLevelInfo;
  }
  
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext(), firstMG, stdLevelInfo)))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstMG, std::string stdLevelInfo) {
  return std::make_unique<PlaceBuffersPass>(firstMG, stdLevelInfo);
}
