#ifndef INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H
#define INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
//#include "dynamatic/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/Logging.h"
//#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "experimental/Support/StdProfiler.h"
#include "dynamatic/Support/LLVM.h"

#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
//#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"

#include <vector>

namespace dynamatic {
namespace experimental {
namespace sharing {

// kosarajus algorithm performed on basic block level
std::vector<int> Kosarajus_algorithm_BBL(SmallVector<ArchBB> archs);

// different implementation: performed on operation level
void Kosarajus_algorithm_OPL(mlir::Operation* startOp, std::set<mlir::Operation*>& result, std::map<Operation *, unsigned int>& topological_sort);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif // INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H
