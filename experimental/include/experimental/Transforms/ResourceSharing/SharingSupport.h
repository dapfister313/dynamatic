#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H


#include "experimental/Transforms/ResourceSharing/FCCM22Sharing.h"
#include "experimental/Transforms/ResourceSharing/SCC.h"
#include "experimental/Transforms/ResourceSharing/modIR.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "mlir/Pass/PassManager.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"
#include "experimental/Transforms/ResourceSharing/SharingFramework.h"

using namespace mlir;
using namespace dynamatic;

//additional files, remove at the end what not needed
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <string>
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm::sys;
using namespace dynamatic::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::sharing;

//std::max
#include <algorithm>
#include <list>

namespace dynamatic {
namespace experimental {
namespace sharing {

//create all possible pairs of Groups in a specific set
inline std::vector<std::pair<GroupIt, GroupIt>> combinations(Set *set) {
    std::vector<std::pair<GroupIt, GroupIt>> result;
    for(GroupIt g1 = set->groups.begin(); g1 != set->groups.end(); g1++) {
        GroupIt g2 = g1;
        g2++;
        for( ; g2 != set->groups.end(); g2++) {
            result.push_back(std::make_pair(g1, g2));
        }
    }
    return result;
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic


namespace dynamatic {
namespace buffer {
namespace fpga20 {

class MyFPGA20Buffers : public FPGA20Buffers {
    public:
    std::vector<ResourceSharingInfo::OperationData> getData();
    LogicalResult addSyncConstraints(std::vector<Value> opaqueChannel);

    //constructor
    MyFPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                            double targetPeriod, bool legacyPlacement)
        : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement){};
};

} // namespace fpga20
} // namespace buffer
} // namespace dynamatic

#endif //EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H