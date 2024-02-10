//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Implements the --resource-sharing pass, which checks for sharable
// Operations (sharable means little to no performance overhead).
//===----------------------------------------------------------------------===//

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
  ResourceSharingFCCM22PerformancePass(ResourceSharing_Data &data, StringRef algorithm,
                        StringRef frequencies, StringRef timingModels,
                        bool firstCFDFC, double targetCP, unsigned timeout,
                        bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        data(data){};

  ResourceSharing_Data &data;

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
  //GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB, double targetPeriod, bool legacyPlacement)
  if (algorithm == "fpga20")
    milp = new fpga20::MyFPGA20Buffers(env, myInfo, timingDB, targetCP,
                                     false);
  else if (algorithm == "fpga20-legacy")
    milp = new fpga20::MyFPGA20Buffers(env, myInfo, timingDB, targetCP,
                                     true);
  milp->addSyncConstraints(data.opaqueChannel);
  assert(milp && "unknown placement algorithm");

  if (failed(milp->optimize()) || failed(milp->getResult(placement)))
    return failure();
 
  data.sharing_feedback.sharing_init = milp->getData();

  controlStructure control_item;
  unsigned int BB_idx = 0;
  // Walkin' in the IR:
  llvm::errs() << "Walkin' inside the IR:\n";
  for (Operation &op : myInfo.funcOp.getOps()) {
    if(op.getName().getStringRef() == "handshake.merge" || op.getName().getStringRef() == "handshake.control_merge") {
      for (const auto &u : op.getResults()) {
        if(u.getType().isa<NoneType>()) {
          BB_idx = getLogicBB(&op).value();
          control_item.control_merge = u;
        }
      }
    }
    if(op.getName().getStringRef() == "handshake.br" || op.getName().getStringRef() == "handshake.cond_br") {
      for (const auto &u : op.getOperands()) {
        if(u.getType().isa<NoneType>()) {
            if(BB_idx != getLogicBB(&op).value()) {
              llvm::errs() << "[critical Error] control channel not present\n";
            }
            control_item.control_branch = u;
            data.control_map[BB_idx] = control_item;
        }
      }
    }
    llvm::errs() << "Operation" << op << "\n";
    std::vector<Operation *> opsToProcess;
    for (auto &u : op.getResults().getUses())
      opsToProcess.push_back(u.getOwner());
    llvm::errs() << "Successors: ";
    for(auto op1 : opsToProcess) {
      llvm::errs() << op1 << ", ";
    }
    llvm::errs() << "\n";
  }

  // If we are in the entry block, we can use the start input of the
  // function (last argument) as our control value
  assert(myInfo.funcOp.getArguments().back().getType().isa<NoneType>() &&
          "expected last function argument to be a NoneType");
  llvm::errs() << "Argument: " << myInfo.funcOp.getArguments().back() << "\n";
  Value func = myInfo.funcOp.getArguments().back();
  std::vector<Operation *> startingOps;
  for (auto &u : func.getUses())
    startingOps.push_back(u.getOwner());
  if(startingOps.size() != 1)
    llvm::errs() << "[Critical Error] Expected 1 starting Operation, got " << startingOps.size() << "\n";
  data.startingOp = startingOps[0];

  for(auto arch_item : myInfo.archs) {
    llvm::errs() << "Source: " << arch_item.srcBB << ", Destination: " << arch_item.dstBB << "\n";
  }

  llvm::errs() << "Setting some random count!\n";
  
  data.archs = myInfo.archs;
  data.funcOp = myInfo.funcOp;

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
} // namespace

void ResourceSharingFCCM22Pass::runDynamaticPass() {
  OpBuilder builder(&getContext());
  llvm::errs() << "***** Resource Sharing *****\n";
  ModuleOp modOp = getOperation();
  ResourceSharing_Data data;


  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();


    // Data object to extract information from buffer placement
    // Use a pass manager to run buffer placement on the current module
    PassManager pm(&getContext());
    pm.addPass(std::make_unique<ResourceSharingFCCM22PerformancePass>(
        data, algorithm, frequencies, timingModels, firstCFDFC, targetCP,
        timeout, dumpLogs));
    //for(int i = 0; i < 3; i++) {
      if (failed(pm.run(modOp))) {
        return signalPassFailure();
      }
    //}

    //NameUniquer names(data.funcOp);
    std::unordered_map<mlir::Operation*, double> data_mod;
    for(auto item : data.sharing_feedback.sharing_init) {
      if (data_mod.find(item.op) != data_mod.end()) {
        data_mod[item.op] = std::max(item.occupancy, data_mod[item.op]);
      } else {
        data_mod[item.op] = item.occupancy;
      }
    }

    initialize_modification(data.control_map);
    revert_to_initial_state();

    ResourceSharing sharing;
    sharing.setFirstOp(data.startingOp);

    //eigther use this
    sharing.initializeTopolocialOpSort();
    //or this
    std::set<mlir::Operation*> ops_with_no_loops;
    sharing.performSCC_opl(ops_with_no_loops);

    sharing.getListOfControlFlowEdges(data.archs);
    int number_of_basic_blocks = sharing.getNumberOfBasicBlocks();
    llvm::errs() << "Number of BBs: " << number_of_basic_blocks << "\n";

    //perform SCC computation
    std::vector<int> SCC = sharing.performSCC_bbl();

    //get number of strongly connected components
    int number_of_SCC = SCC.size();

    sharing.retrieveDataFromPerformanceAnalysis(data.sharing_feedback, SCC, number_of_SCC, timingDB);
    sharing.print();
   // iterating over different operation types
   for(auto& op_type : sharing.operation_types) {
    // Sharing within a loop nest
    for(auto& set : op_type.sets) {
      bool groups_modified = true;
      while(groups_modified) {
        groups_modified = false;

        //iterate over combinations of groups
        for(auto pair : combinations(&set)) {
          //check if sharing is potentially possible
          double occupancy_sum = pair.first->shared_occupancy + pair.second->shared_occupancy;
          //change to separate function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          if(occupancy_sum <= op_type.op_latency) {
            std::vector<Operation*> finalOrd;
            //check if operations on loop
            if(!pair.first->hasCycle && !pair.second->hasCycle) {
              llvm::errs() << "[comp] Non-cyclic\n";
              finalOrd = sharing.sortTopologically(pair.first, pair.second);
              if(!sharing.isTopologicallySorted(finalOrd)) {
                llvm::errs() << "[info] Failed topological sorting\n";
              }
              groups_modified = true;
            } else {
              llvm::errs() << "[comp] Cyclic\n";
              // Search for best group ordering
              std::vector<Operation*> current_permutation;
              current_permutation.insert(current_permutation.end(), pair.first->items.begin(), pair.first->items.end());
              current_permutation.insert(current_permutation.end(), pair.second->items.begin(), pair.second->items.end());
              std::sort(current_permutation.begin(), current_permutation.end());
              //seperate function for permutations!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              do {
                //Print out current permutation
                llvm::errs() << "[Permutation] Start\n";
                for(auto op : current_permutation) {
                  llvm::errs() << op << ", ";
                }
                llvm::errs() << "\n";

                //run_performance analysis here !!!!!!!!!!!!!!!!!!!
                generate_performance_model(&builder, current_permutation);
                deleteAllBuffers(data.funcOp);
                if (failed(pm.run(modOp))) {
                  return signalPassFailure();
                }
                destroy_performance_model(&builder, current_permutation);
                //check if no performance loss, if yes, break
                ResourceSharing temp_sharing;
                temp_sharing.retrieveDataFromPerformanceAnalysis(data.sharing_feedback, SCC, number_of_SCC, timingDB);
                if(true) {
                  finalOrd = current_permutation;
                  break;
                }
              } while (next_permutation (current_permutation.begin(), current_permutation.end()));
            }
            if(finalOrd.size() != 0) {
                //Merge groups, update ordering and update shared occupancy
                set.joinGroups(pair.first, pair.second, finalOrd);
                break;
            }
          }
        }
      }
    }
    op_type.print();
    // Sharing across loop nests
    op_type.sharingAcrossLoopNests();

    op_type.printFinalGroup();

    // Sharing other units
    op_type.sharingOtherUnits();

    op_type.printFinalGroup();

    //here we want to create the performance model
    //auto op = op_type.final_grouping.groups.begin()->items[0];
    std::vector<Value> return_values = {};
    for (auto op :
      llvm::make_early_inc_range(data.funcOp.getOps<arith::MulIOp>())) {
        return_values.push_back(generate_performance_step(&builder, op));
      }

    data.opaqueChannel = return_values;
    deleteAllBuffers(data.funcOp);
    if (failed(pm.run(modOp))) {
      return signalPassFailure();
    }
    //dynamatic::sharing::revert_performance_step(&builder, op);
    break;
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
