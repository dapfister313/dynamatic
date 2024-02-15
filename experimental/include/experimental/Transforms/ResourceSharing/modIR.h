#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_MODIR_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_MODIR_H

#include "dynamatic/Support/CFG.h"

namespace dynamatic {
namespace experimental {
namespace sharing {

struct controlStructure {
    mlir::Value control_merge;
    mlir::Value control_branch;
    mlir::Value current_position;
};

std::vector<Value> generate_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items, std::map<int, controlStructure>& control_map);
void destroy_performance_model(OpBuilder* builder, std::vector<mlir::Operation*>& items);

Value generate_performance_step(OpBuilder* builder, mlir::Operation *op, std::map<int, controlStructure>& control_map);
void revert_performance_step(OpBuilder* builder, mlir::Operation *op);

mlir::OpResult add_fork(OpBuilder* builder, mlir::OpResult connectionPoint);

mlir::OpResult extend_fork(OpBuilder* builder, handshake::ForkOp OldFork);

void addSink(OpBuilder* builder, mlir::OpResult* connectionPoint);

mlir::OpResult addConst(OpBuilder* builder, mlir::OpResult* connectionPoint, int value);

mlir::OpResult addBranch(OpBuilder* builder, mlir::OpResult* connectionPoint);

mlir::OpResult addMerge(OpBuilder* builder, mlir::OpResult* connectionPoint);

void deleteAllBuffers(handshake::FuncOp funcOp);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic



#endif // EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_MODIR_H
