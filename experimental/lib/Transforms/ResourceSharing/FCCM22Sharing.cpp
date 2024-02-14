//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Implements the --resource-sharing pass, which checks for sharable
// Operations (sharable means little to no performance overhead).
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/NameUniquer.h"
//#include "dynamatic/Support/NameUniquer.h"
#include "experimental/Transforms/ResourceSharing/FCCM22Sharing.h"
#include "experimental/Transforms/ResourceSharing/SCC.h"
#include "experimental/Transforms/ResourceSharing/modIR.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "mlir/Pass/PassManager.h"
#include "dynamatic/Support/DynamaticPass.h"
//#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"
//#include "dynamatic/Support/LogicBB.h"
#include "experimental/Transforms/ResourceSharing/SharingSupport.h"
#include "experimental/Transforms/ResourceSharing/SharingFramework.h"

using namespace mlir;
using namespace dynamatic;

//additional files, remove at the end what not needed
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
//#include "dynamatic/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/Logging.h"
//#include "dynamatic/Support/LogicBB.h"
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
//#include "dynamatic/Conversion/StandardToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
//#include "dynamatic/Dialect/Handshake/HandshakePasses.h"
//#include "dynamatic/Dialect/Pipeline/Pipeline.h"
//#include "dynamatic/Support/BackedgeBuilder.h"
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

namespace {

struct ResourceSharingFCCM22PerformancePass : public HandshakePlaceBuffersPass {
  ResourceSharingFCCM22PerformancePass(ResourceSharingInfo &data, StringRef algorithm,
                        StringRef frequencies, StringRef timingModels,
                        bool firstCFDFC, double targetCP, unsigned timeout,
                        bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        data(data){};

  ResourceSharingInfo &data;

protected:
  
  LogicalResult getBufferPlacement(FuncInfo &info, TimingDatabase &timingDB, 
                                   Logger *logger, BufferPlacement &placement) override;
};

} //namespace

LogicalResult ResourceSharingFCCM22PerformancePass::getBufferPlacement(
    FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
    BufferPlacement &placement) {
      FuncInfo myInfo = info;
  
  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  // Create and solve the MILP
  fpga20::MyFPGA20Buffers *milp = nullptr;
  
  if (algorithm == "fpga20")
    milp = new fpga20::MyFPGA20Buffers(env, myInfo, timingDB, targetCP,
                                     false, *logger, "1");
  else if (algorithm == "fpga20-legacy")
    milp = new fpga20::MyFPGA20Buffers(env, myInfo, timingDB, targetCP,
                                     true, *logger, "1");
  
  assert(milp && "unknown placement algorithm");

  if(failed(milp->addSyncConstraints(data.opaqueChannel))) {
    return failure();
  }

  if (failed(milp->optimize()) || failed(milp->getResult(placement)))
    return failure();
  
  data.funcOp = myInfo.funcOp;

  if(data.fullReportRequired) {
    data.operations = milp->getData();
    data.archs = myInfo.archs;
  } else { 
    data.occupancySum = milp->getOccupancySum(data.testedGroups);
  }

  delete milp;
  return success();
}

namespace {

struct ResourceSharingFCCM22Pass
    : public dynamatic::experimental::sharing::impl::ResourceSharingFCCM22Base<
          ResourceSharingFCCM22Pass> {

  ResourceSharingFCCM22Pass(StringRef algorithm, StringRef frequencies,
                                StringRef timingModels, bool firstCFDFC,
                                double targetCP, unsigned timeout,
                                bool dumpLogs) {
    this->algorithm = algorithm.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
    this->firstCFDFC = firstCFDFC;
    this->targetCP = targetCP;
    this->timeout = timeout;
    this->dumpLogs = dumpLogs;
  }

  void runDynamaticPass() override;
};

bool runPerformanceAnalysisOfOnePermutation(ResourceSharingInfo &data, std::vector<Operation*>& current_permutation,
                                            ResourceSharing& sharing, OpBuilder* builder, PassManager& pm, ModuleOp& modOp) {
    deleteAllBuffers(data.funcOp);
    data.opaqueChannel = generate_performance_model(builder, current_permutation, sharing.control_map);
    if (failed(pm.run(modOp))) {
        return false;
    }
    destroy_performance_model(builder, current_permutation);
    return true;
}

bool runPerformanceAnalysis(GroupIt group1, GroupIt group2, double occupancy_sum, ResourceSharingInfo &data, OpBuilder* builder, 
                            PassManager& pm, ModuleOp& modOp, std::vector<Operation*>& finalOrd, ResourceSharing& sharing) {
    // Search for best group ordering
    std::vector<Operation*> current_permutation;
    current_permutation.insert(current_permutation.end(), group1->items.begin(), group1->items.end());
    current_permutation.insert(current_permutation.end(), group2->items.begin(), group2->items.end());
    data.testedGroups.clear();
    std::copy(current_permutation.begin(), current_permutation.end(), std::inserter(data.testedGroups, data.testedGroups.end()));
    std::sort(current_permutation.begin(), current_permutation.end());

    do {
        if(runPerformanceAnalysisOfOnePermutation(data, current_permutation,sharing, builder, pm, modOp) == false) {
          return false;
        }
        if(equal(occupancy_sum, data.occupancySum)) {
            finalOrd = current_permutation;
            break;
        }
    } while (next_permutation (current_permutation.begin(), current_permutation.end()));
    return true;
}

} // namespace

void ResourceSharingFCCM22Pass::runDynamaticPass() {
  llvm::errs() << "***** Resource Sharing *****\n";
  OpBuilder builder(&getContext());
  ModuleOp modOp = getOperation();
  
  ResourceSharingInfo data;
  data.fullReportRequired = true;

  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();

  // running buffer placement on current module
  PassManager pm(&getContext());
  pm.addPass(std::make_unique<ResourceSharingFCCM22PerformancePass>(
      data, algorithm, frequencies, timingModels, firstCFDFC, targetCP,
      timeout, dumpLogs));
  if (failed(pm.run(modOp))) {
      return signalPassFailure();
  }
  
  // placing data retrieved from buffer placement
  ResourceSharing sharing(data, timingDB);

  // delete !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  NameUniquer names(data.funcOp);

  // from now on we are only interested in the occupancy sum
  data.fullReportRequired = false;
  
  // iterating over different operation types
  for(auto& op_type : sharing.operation_types) {
    // Sharing within a loop nest
    for(auto& set : op_type.sets) {
      bool groups_modified = true;
      while(groups_modified) {
        groups_modified = false;
        std::vector<std::pair<GroupIt, GroupIt>> combination = combinations(&set);
        //iterate over combinations of groups
        for(auto pair : combination) {
          //check if sharing is potentially possible
          double occupancy_sum = pair.first->shared_occupancy + pair.second->shared_occupancy;
          
          //change to separate function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          if(lessOrEqual(occupancy_sum, op_type.op_latency)) {
            std::vector<Operation*> finalOrd;
            //check if operations on loop
            if(!pair.first->hasCycle && !pair.second->hasCycle) {
              finalOrd = sharing.sortTopologically(pair.first, pair.second);
            } else {
              runPerformanceAnalysis(pair.first, pair.second, occupancy_sum, data, &builder, 
                                     pm, modOp, finalOrd, sharing);
            }
            if(finalOrd.size() != 0) {
                //Merge groups, update ordering and update shared occupancy
                set.joinGroups(pair.first, pair.second, finalOrd);
                groups_modified = true;
                break;
            }
          }
        }
      }
    }
    
    // Sharing across loop nests
    op_type.sharingAcrossLoopNests();

    // Sharing other units
    op_type.sharingOtherUnits();
    
    // print final grouping
    op_type.printFinalGroup();
  }
}

namespace dynamatic {
namespace experimental {
namespace sharing {

/// Returns a unique pointer to an operation pass that matches MLIR modules.
std::unique_ptr<dynamatic::DynamaticPass>
createResourceSharingFCCM22Pass(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<ResourceSharingFCCM22Pass>(
      algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs);
}
} // namespace sharing
} // namespace experimental
} // namespace dynamatic
