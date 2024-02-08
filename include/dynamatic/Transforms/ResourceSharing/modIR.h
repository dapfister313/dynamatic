#ifndef INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_MODIR_H
#define INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_MODIR_H

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"

#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"
namespace dynamatic {
namespace sharing {
struct controlStructure {
    mlir::Value control_merge;
    mlir::Value control_branch;
    mlir::Value current_position;
};
std::optional<unsigned> getLogicBB(Operation *op);

void initialize_modification(std::map<int, controlStructure> control_map);
void revert_to_initial_state();

std::vector<Value> generate_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items);
void destroy_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items);

Value generate_performance_step(OpBuilder* builder, mlir::Operation *op);
void revert_performance_step(OpBuilder* builder, mlir::Operation *op);

mlir::OpResult add_fork(OpBuilder* builder, mlir::OpResult connectionPoint);

mlir::OpResult extend_fork(OpBuilder* builder, ForkOp OldFork);

void addSink(OpBuilder* builder, mlir::OpResult* connectionPoint);

mlir::OpResult addConst(OpBuilder* builder, mlir::OpResult* connectionPoint, int value);

mlir::OpResult addBranch(OpBuilder* builder, mlir::OpResult* connectionPoint);

mlir::OpResult addMerge(OpBuilder* builder, mlir::OpResult* connectionPoint);

void deleteAllBuffers(FuncOp funcOp);
}
}


#endif // INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_MODIR_H