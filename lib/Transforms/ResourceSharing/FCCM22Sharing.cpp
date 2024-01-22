//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Implements the --resource-sharing pass, which checks for sharable
// Operations (little to none performance overhead).
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ResourceSharing/FCCM22Sharing.h"
#include "dynamatic/Transforms/ResourceSharing/SCC.h"
#include "dynamatic/Transforms/ResourceSharing/modIR.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"
#include "dynamatic/Support/LogicBB.h"

using namespace mlir;
using namespace circt;

//additional files
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
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
#include <string>
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Pipeline/Pipeline.h"
#include "circt/Support/BackedgeBuilder.h"
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
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

//std::max
#include <algorithm>

namespace {

std::optional<unsigned> getLogicBB(Operation *op) {
  if (auto bb = op->getAttrOfType<mlir::IntegerAttr>(BB_ATTR))
    return bb.getUInt();
  return {};
}

static void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (int i = 0, e = op->getNumOperands(); i < e; ++i)
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      break;
    }
  return;
}

/// Recovered data needed for performing resource sharing
struct ResourceSharing_Data {
  //extracts needed resource sharing data from FuncInfo struct
  ResourceSharingInfo sharing_feedback; 
  //used to perform SCC-computation (finding strongly connected components)
  SmallVector<experimental::ArchBB> archs;

  unsigned someCountOfSomething = 0;
  unsigned totalNumberOfOpaqueBuffers = 0;
};

struct ResourceSharingFCCM22PerformancePass : public HandshakePlaceBuffersPass {
  ResourceSharingFCCM22PerformancePass(ResourceSharing_Data &data, StringRef algorithm,
                        StringRef frequencies, StringRef timingModels,
                        bool firstCFDFC, double targetCP, unsigned timeout,
                        bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        data(data){};

  /// Some data you care about extracting.
  ResourceSharing_Data &data;

protected:
  /// Custom buffer placement step. Copied from `HandshakePlaceBuffersPass` with
  /// the addition of a step at the end to extract some information from the
  /// MILP.
  LogicalResult
  getBufferPlacement(FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
                     DenseMap<Value, PlacementResult> &placement) override;
};

/*
       Inside each set of strongly connected components
       one needs to check if sharing is possible
*/
struct Group {
  std::vector<mlir::Operation*> items;
  double shared_occupancy;

  //Constructors
  Group(mlir::Operation* op, double occupancy)
        : shared_occupancy(occupancy) {
          items.push_back(op);
        }
  Group(mlir::Operation* op)
        : shared_occupancy(-1) {
          items.push_back(op);
        }


  //Destructor
  ~Group() {};
};

/*
       Each building block resides in a set
       of strongly connected components
*/
struct Set {
  std::list<Group> groups{};
  int SCC_id;
  double op_latency;

  void addGroup(Group group) {
    groups.push_back(group);
  }

  //Constructor
  Set(Group group) {
    groups.push_back(group);
  }

  Set(int SCC_idx, double latency) {
    SCC_id = SCC_idx;
    op_latency = latency;
  }
  
  /*
  bool groups_mergeable(int i, int j) {
    if(i == j) {
      return false;
    }

    if(groups[i].occupancy + groups[j].occupancy > op_latency) {
      return false;
    }
    
    //make performance analysis here

    return true;
  }

  bool merge_groups(int i, int j) {
    if(!groups_mergeable(i, j)) {
      return false;
    }

    //merge groups here

    return true;
  }
  */
};

/*
       Each operation type (e.g. mul, add, load) 
       can be treated separately
*/
struct OpSelector {
  double op_latency;
  llvm::StringRef identifier;
  std::vector<Set> sets{};
  std::map<int, int> SetSelect;
  
  void addSet(Group group) {
    sets.push_back(Set(group));
  }

  //Constructor
  OpSelector(double latency, llvm::StringRef identifier)
        : op_latency(latency), identifier(identifier) {}

};

/*
       Class to iterate easily trough all 
       operation types
*/
class ResourceSharing {
  std::vector<OpSelector> operation_type; //{};
  std::map<int, double> throughput;
  SmallVector<experimental::ArchBB> archs;
  std::map<llvm::StringRef, int> OpNames;
  int number_of_operation_types;
  
  double runPerformanceAnalysis() {
    return 0;
  }

public:
  //place resource sharing data retrieved from buffer placement
  void place_data(ResourceSharingInfo sharing_feedback, std::vector<int>& SCC, int number_of_SCC, TimingDatabase timingDB) {
    //Take biggest occupancy per operation
    std::unordered_map<mlir::Operation*, std::pair<double,double>> data_mod;
    for(auto item : sharing_feedback.sharing_init) {
      if (data_mod.find(item.op) != data_mod.end()) {
        data_mod[item.op].first = std::max(item.occupancy, data_mod[item.op].first);
      } else {
        data_mod[item.op] = std::make_pair(item.occupancy, item.op_latency);
      }
    }

    //everytime we place/overwrite data, initial number of operation types is 0;
    number_of_operation_types = 0;
    
    //iterate through all retrieved operations
    for(auto sharing_item : data_mod) {
      //choose the right operation type
      double latency;
      if (failed(timingDB.getLatency(sharing_item.first, latency)))
        latency = 0.0;
      llvm::errs() << "Latency of unit " << sharing_item.first << ": " << latency << "\n";
      llvm::StringRef OpName = sharing_item.first->getName().getStringRef();
      Group group_item = Group(sharing_item.first, sharing_item.second.first);
      int OpIdx = -1;
      auto item = OpNames.find(OpName);
      if(item != OpNames.end()) {
        OpIdx = item->second;
      } else {
        OpNames[OpName] = number_of_operation_types;
        OpIdx = number_of_operation_types;
        ++number_of_operation_types;
        operation_type.push_back(OpSelector(sharing_item.second.second, OpName));
      }
      OpSelector& OpT = operation_type[OpIdx];
      
      //choose the right set
      int SetIdx = -1;
      unsigned int BB = getLogicBB(sharing_item.first).value();
      int SCC_idx = SCC[BB];
      auto set_select = OpT.SetSelect.find(SCC_idx);
      if(set_select != OpT.SetSelect.end()) {
        SetIdx = set_select->second;
      } else {
        SetIdx = OpT.SetSelect.size();
        OpT.SetSelect[SCC_idx] = SetIdx;
        OpT.sets.push_back(Set(SCC_idx, latency));
      }
      Set& SetT = OpT.sets[SetIdx];

      //Simply add group to set
      SetT.groups.push_front(group_item);
    }
    throughput = sharing_feedback.sharing_check;
  }

  int place_BB(SmallVector<experimental::ArchBB> archs_ext) {
    archs = archs_ext;
    unsigned int maximum = 0;
    for(auto arch_item : archs) {
      maximum = std::max(maximum, std::max(arch_item.srcBB, arch_item.dstBB));
    }
    return maximum + 1; //as we have BB0, we need to add one at the end
  }
  /*
  void recursiveDFS(unsigned int starting_node, std::vector<bool> &visited) {
    visited[starting_node] = true;
    for(auto arch_item : archs) {
      if(arch_item.srcBB == starting_node && !visited[arch_item.dstBB]) {
        recursiveDFS(arch_item.dstBB, visited);
      }
    }
  }

  int max(int a, int b) {
    if(a > b) {
      return a;
    }
    return b;
  }
  
  int find_strongly_connected_components(std::vector<int>& SCC) {
    int BBs = 0;
    for(auto arch_item : archs) {
      BBs = max(BBs, arch_item.srcBB);
      BBs = max(BBs, arch_item.dstBB);
    }
    BBs += 1;
    llvm::errs() << "Number of BBs: " << BBs << "\n\n";

    std::vector<std::vector<bool>> visited(BBs, std::vector<bool>(BBs, false));
    
    for(int i = 0; i < BBs; i++) {
      recursiveDFS(i, visited[i]);
    }

    std::vector<int> Sets(BBs);
    int position = 1;
    bool taken = false;
    std::vector<int> num_of_items(BBs);

    for(int i = 0; i < BBs; i++) {
      for(int j = 0; j < BBs; j++) {
        num_of_items[i] += visited[i][j];
      }
    }
    
    for(int i = 0; i <= BBs; i++) {
      for(int j = 0; j < BBs; j++) {
        if(num_of_items[j] != i) {
          continue;
        }
        for(int k = 0; k < BBs; k++) {
          if(visited[j][k] && !Sets[k]) {
            Sets[k] = position;
            taken = true;
          }
        }
      }
      if(taken) {
        position += 1;
        taken = false;
      }
    }
    for(int i = 0; i < BBs; i++) {
      SCC[i] = Sets[i] - 1;
    }
    llvm::errs() << "\n\n";
    return position - 1;
  }
  */

  std::vector<int> performSCC_bbl() {
    return Kosarajus_algorithm_BBL(archs);
  }
  
  void print() {
    llvm::errs() << "\n***** Basic Blocks *****\n";
    for(auto arch_item : archs) {
      llvm::errs() << "Source: " << arch_item.srcBB << ", Destination: " << arch_item.dstBB << "\n";
    }
    std::map<int, double>::iterator it = throughput.begin();
    llvm::errs() << "\n**** Throughput per CFDFC ****\n";
    for(; it != throughput.end(); it++) {
      llvm::errs() << "CFDFC #" << it->first << ": " << it->second << "\n";
    }
    for(auto Op : operation_type) {
      llvm::errs() << "\n*** New Operation type: " << Op.identifier << " ***\n";
      for(auto set : Op.sets) {
        llvm::errs() << "** New set **\n";
        for(auto group : set.groups) {
          llvm::errs() << "* New group *\n";
          llvm::errs() << "Number of entries: " << group.items.size() << "\n";
        }
      }
    }
  }
};

} //namespace

LogicalResult ResourceSharingFCCM22PerformancePass::getBufferPlacement(
    FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
    DenseMap<Value, PlacementResult> &placement) {
  /// This is exactly the same as the getBufferPlacement method in
  /// HandshakePlaceBuffersPass

  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  // Create and solve the MILP
  BufferPlacementMILP *milp = nullptr;
  if (algorithm == "fpga20")
    milp = new fpga20::FPGA20Buffers(info, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, false);
  else if (algorithm == "fpga20-legacy")
    milp = new fpga20::FPGA20Buffers(info, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, true);
  assert(milp && "unknown placement algorithm");
  int milpStat;
  LogicalResult res = success();
  if (failed(milp->optimize(&milpStat))) {
    res = info.funcOp->emitError()
          << "Buffer placement MILP failed with status " << milpStat
          << ", reason:" << getGurobiOptStatusDesc(milpStat);
  } else if (failed(milp->getPlacement(placement))) {
    res = info.funcOp->emitError()
          << "Failed to extract placement decisions from MILP's solution.";
  }
  //std::vector<Op_stats> occupancy_info; //!!!!!
  // Walkin' in the IR:
  
  //value after start is "info.funcOp->getArguments().back()"
  /*
  for(auto result : info.funcOp->getArguments().back()) {
    llvm::errs() << "Argument: " << result << "\n";
  }
  */
  // If we are in the entry block, we can use the start input of the
  // function (last argument) as our control value
  assert(info.funcOp.getArguments().back().getType().isa<NoneType>() &&
          "expected last function argument to be a NoneType");
  llvm::errs() << "Argument: " << info.funcOp.getArguments().back() << "\n";
  
  /*
  unsigned int number_of_basic_blocks = 0;
  for(auto arch_item : info.archs) {
    number_of_basic_blocks = std::max(number_of_basic_blocks, std::max(arch_item.srcBB, arch_item.dstBB));
  }
  ++number_of_basic_blocks;

  llvm::errs() << "Number of basic blocks: " << number_of_basic_blocks << "\n";
  */

  SmallVector<mlir::Block *> logicBBconversion;
  int currentHighestBB = -1;
  llvm::errs() << "Walkin' inside the IR:\n";
  for (Operation &op : info.funcOp.getOps()) {
    llvm::errs() << "Operation" << op << "\n";
    std::optional<unsigned> brBB = getLogicBB(&op);
    if(brBB) {
      if((unsigned)(currentHighestBB + 1) == brBB.value()) {
        logicBBconversion.push_back(&op.getParentRegion()->getBlocks().front());
        ++currentHighestBB;
      }
      //llvm::errs() << "Logic BB: " << brBB.value() << "\n";
    }
  }

  // Here, before destroying the MILP, extract whatever information you want
  // and store it into your MyData& reference. If you need to extract variable
  // values from the MILP you may need to make some of its fields public (to
  // be discussed in PRs).
  llvm::errs() << "Setting some random count!\n";
  data.someCountOfSomething += 10;
  llvm::errs() << "Current count: " << data.someCountOfSomething << "\n";

  data.sharing_feedback = info.sharing_info;
  data.archs = info.archs;
  
  /*
  for(auto arch_item : info.archs) {
    llvm::errs() << "Source: " << arch_item.srcBB << ", Destination: " << arch_item.dstBB << "\n";
  }
  */
  
  /*
  OpBuilder builder(&getContext());
  llvm::errs() << "Test!" << "\n";
  std::vector<circt::handshake::ForkOp *> delete_vector;
  int break_point = 0;

  for (auto forkToReplace :
      llvm::make_early_inc_range(info.funcOp.getOps<ForkOp>())) {
    if(break_point == 3) {
      llvm::errs() << forkToReplace << "\n";
      Operation *opSrc = forkToReplace.getOperand().getDefiningOp();
      Value opSrcIn = forkToReplace.getOperand();
      std::vector<Operation *> opsToProcess;
      for (auto &u : forkToReplace.getResults().getUses())
        opsToProcess.push_back(u.getOwner());
    
      // Insert fork after op
      builder.setInsertionPointAfter(opSrc);
      auto forkSize = opsToProcess.size();
      llvm::errs() << "Fork Size: " << forkSize << "************************************\n";
      
      auto newForkOp = builder.create<ForkOp>(opSrcIn.getLoc(), opSrcIn, forkSize + 1);
      inheritBB(opSrc, newForkOp);
      for (int i = 0, e = forkSize; i < e; ++i)
        opsToProcess[i]->replaceUsesOfWith(forkToReplace->getResult(i), newForkOp->getResult(i));
      auto newSinkOp = builder.create<SinkOp>(newForkOp->getResult(forkSize).getLoc(), newForkOp->getResult(forkSize));
      forkToReplace.erase();
      
      break;
    }
    ++break_point;
  }
  */
  /*
  OpBuilder builder(&getContext());
  llvm::errs() << "Test!" << "\n";
  int break_point = 0;
  for (auto forkToReplace :
      llvm::make_early_inc_range(info.funcOp.getOps<ForkOp>())) {
        if(break_point) {
          llvm::errs() << "Fork: " << forkToReplace << "\n";
          Operation *opSrc = forkToReplace.getOperand().getDefiningOp();
          Value opSrcIn = forkToReplace.getOperand();
          std::vector<Operation *> opsToProcess;
          for (auto &u : forkToReplace.getResults().getUses())
            opsToProcess.push_back(u.getOwner());
        
          // Insert fork after op
          builder.setInsertionPointAfter(opSrc);
          auto forkSize = opsToProcess.size();
          llvm::errs() << "Fork Size: " << forkSize << "************************************\n";
          
          auto newForkOp = builder.create<ForkOp>(opSrcIn.getLoc(), opSrcIn, forkSize + 2);
          inheritBB(opSrc, newForkOp);
          for (int i = 0, e = forkSize; i < e; ++i)
            opsToProcess[i]->replaceUsesOfWith(forkToReplace->getResult(i), newForkOp->getResult(i));
          Value condValue = newForkOp->getResult(forkSize); //newForkOp.getOperand(); //dyn_cast<mlir::cf::CondBranchOp>(newForkOp).getCondition();
          llvm::errs() << "Resulted value: " << condValue << "\n";
          llvm::errs() << "Previous value: " << forkToReplace->getResult(forkSize-1) << "\n";
          auto newCbranch = builder.create<ConditionalBranchOp>(newForkOp->getResult(forkSize).getLoc(), newForkOp->getResult(forkSize), newForkOp->getResult(forkSize + 1));
          //auto newSinkOp = builder.create<SinkOp>(newForkOp_red->getResult(0).getLoc(), newForkOp_red->getResult(0));
          for(int i = 0; i < 2; i++) {
            auto newSinkOp = builder.create<SinkOp>(newCbranch->getResult(i).getLoc(), newCbranch->getResult(i));
          }
          forkToReplace.erase();
          break;
        }
        break_point++;
  }
  */
  /*
  OpBuilder builder(&getContext());
  for (auto cmpi_to_add : llvm::make_early_inc_range(info.funcOp.getOps<arith::CmpIOp>())) {
    Operation *opSrc = cmpi_to_add;
    Operation *opDst = cmpi_to_add->getNextNode();
    builder.setInsertionPointAfter(opSrc);

    Value bufferIn = opDst->getOperand(0);
    auto placeBuffer = [&](BufferTypeEnum bufType, unsigned numSlots) {
      if (numSlots == 0)
        return;

      // Insert an opaque buffer
      auto bufOp = builder.create<handshake::BufferOp>(
          bufferIn.getLoc(), bufferIn, numSlots, bufType);
      inheritBB(opSrc, bufOp);
      Value bufferRes = bufOp.getResult();

      opDst->replaceUsesOfWith(bufferIn, bufferRes);
      bufferIn = bufferRes;
    };
    break;
  };
  */
  
  /*
  OpBuilder builder(&getContext());
  llvm::errs() << "Test!" << "\n";
  int break_point = 0;
  for (auto cmpi_to_add : llvm::make_early_inc_range(info.funcOp.getOps<arith::CmpIOp>())) {
    Operation *opSrc = cmpi_to_add;
    builder.setInsertionPointAfter(opSrc);
    llvm::errs() << "Operation source: " << opSrc->getName().getStringRef() << "\n";
    Operation *opDst = cmpi_to_add->getNextNode();
    llvm::errs() << "Operation destination: " << opDst->getName().getStringRef() << "\n";
    Value opDstOut = opDst->getOperand(0);
    Value opSrcIn = opSrc->getResult(0);
    auto newForkOp = builder.create<ForkOp>(opSrcIn.getLoc(), opSrcIn, 3);
    inheritBB(cmpi_to_add, newForkOp);
    Value OpRes = newForkOp->getResult(0);
    opDst->replaceUsesOfWith(opDstOut, OpRes);
    //auto newSrc = builder.create<SourceOp>(...);
    auto newCbranch = builder.create<ConditionalBranchOp>(newForkOp->getResult(1).getLoc(), newForkOp->getResult(1), newForkOp->getResult(2));
    for(int i = 0; i < 2; i++) {
      auto newSinkOp = builder.create<SinkOp>(newCbranch->getResult(i).getLoc(), newCbranch->getResult(i));
    }
    break;
  }
  */
  /*
  //perform merge
  OpBuilder builder(&getContext());
  for (auto forkToReplace :
      llvm::make_early_inc_range(info.funcOp.getOps<ForkOp>())) {

      llvm::errs() << forkToReplace << "\n";
      Operation *opSrc = forkToReplace.getOperand().getDefiningOp();
      Value opSrcIn = forkToReplace.getOperand();
      std::vector<Operation *> opsToProcess;
      for (auto &u : forkToReplace.getResults().getUses())
        opsToProcess.push_back(u.getOwner());
    
      // Insert fork after op
      builder.setInsertionPointAfter(opSrc);
      auto forkSize = opsToProcess.size();
      
      auto newForkOp = builder.create<ForkOp>(opSrcIn.getLoc(), opSrcIn, forkSize + 1);
      inheritBB(opSrc, newForkOp);
      for (int i = 0, e = forkSize; i < e; ++i)
        opsToProcess[i]->replaceUsesOfWith(forkToReplace->getResult(i), newForkOp->getResult(i));
      
      auto newSourceOp = builder.create<handshake::SourceOp>(newForkOp.getLoc(), builder.getNoneType());
      IntegerAttr cond = builder.getBoolAttr(true);
      auto newConstOp = builder.create<handshake::ConstantOp>(newSourceOp.getLoc(), cond.getType(), cond, newSourceOp);
      auto newConstOp_2 = builder.create<handshake::ConstantOp>(newForkOp->getResult(forkSize).getLoc(), cond.getType(), cond, newForkOp->getResult(forkSize));
      auto newCbranch = builder.create<ConditionalBranchOp>(newConstOp_2.getLoc(), newConstOp_2, newConstOp);
      inheritBB(opSrc, newCbranch);
      SmallVector<Value> ForkOpS;
      for(int i = 0; i < 2; i++) {
        auto newForkOp2 = builder.create<ForkOp>(newCbranch->getResult(i).getLoc(), newCbranch->getResult(i), 1);
        ForkOpS.push_back(newForkOp2->getResult(0));
      }
      //MergeOp: create merge here! 
      auto newMergeOp = builder.create<handshake::MergeOp>(ForkOpS[0].getLoc(), ForkOpS);

      auto newSourceOp2 = builder.create<handshake::SourceOp>(newForkOp.getLoc(), builder.getNoneType());
      //auto newConstOp3 = builder.create<handshake::ConstantOp>(newSourceOp2.getLoc(), cond.getType(), cond, newSourceOp2);
      auto newSourceOp3 = builder.create<handshake::SourceOp>(newForkOp.getLoc(), builder.getNoneType());
      IntegerAttr cond2 = builder.getBoolAttr(false);
      auto newConstOp4 = builder.create<handshake::ConstantOp>(newSourceOp2.getLoc(), cond2.getType(), cond2, newSourceOp3);
      auto newCbranch2 = builder.create<ConditionalBranchOp>(newSourceOp2.getLoc(), newSourceOp2, newConstOp4);
      for(int i = 0; i < 2; i++) {
        auto newSinkOp = builder.create<SinkOp>(newCbranch2->getResult(i).getLoc(), newCbranch2->getResult(i));
      }
      newSourceOp2->getResult(0).replaceAllUsesWith(newMergeOp.getResult());
      //newConstOp3->getResult(0).replaceUsesOfWith(newMergeOp.getResult());

      
      //newSinkOp->getOperand(0).replaceAllUsesWith(newForkOp->getResult(forkSize));
      forkToReplace.erase();
      //newSourceOp2->getResult(0);
      newSourceOp2.erase();
      //newConstOp3.erase();
      break;
  }
  */
  
  
  OpBuilder builder(&getContext());
  mlir::OpResult connectionPoint;
  for (auto forkToReplace : llvm::make_early_inc_range(info.funcOp.getOps<ForkOp>())) {
    connectionPoint = extend_fork(&builder, forkToReplace);
    break;
  }
  connectionPoint = addConst(&builder, &connectionPoint, 0);
  connectionPoint = addBranch(&builder, &connectionPoint);
  addSink(&builder, &connectionPoint);
  
  /*
  for(auto block : logicBBconversion) {
    llvm::errs() << "Current block: " << *block << "*****************************************************************\n";
    
    for (mlir::Operation &op : block->getOperations()) {
        llvm::errs() << "Traversing operation " << op << "\n";
    }
    
  }
  */

  llvm::errs() << "CHECK POINT: This should be visible!\n";
  data.someCountOfSomething += 20;
  delete milp;
  return res;
}

namespace { 

struct ResourceSharingFCCM22Pass
    : public dynamatic::sharing::impl::ResourceSharingFCCM22Base<
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

  void runOnOperation() override;
};
} // namespace

void ResourceSharingFCCM22Pass::runOnOperation() {
  llvm::errs() << "***** Resource Sharing *****\n";
  ModuleOp modOp = getOperation();
  ResourceSharing_Data data;
  
  while (true) {
    // Data object to extract information from buffer placement
    // Use a pass manager to run buffer placement on the current module
    PassManager pm(&getContext());
    pm.addPass(std::make_unique<ResourceSharingFCCM22PerformancePass>(
        data, algorithm, frequencies, timingModels, firstCFDFC, targetCP,
        timeout, dumpLogs));
    if (failed(pm.run(modOp))) {
      return signalPassFailure();
    }
    /*

    llvm::errs() << "\nInitally:\n";
    
    for(auto item : data.sharing_feedback.sharing_init) {
      item.print();
    }

    llvm::errs() << "\nAfter modification:\n";
    
    std::unordered_map<mlir::Operation*, double> data_mod;
    for(auto item : data.sharing_feedback.sharing_init) {
      if (data_mod.find(item.op) != data_mod.end()) {
        data_mod[item.op] = std::max(item.occupancy, data_mod[item.op]);
      } else {
        data_mod[item.op] = item.occupancy;
      }
    }
    
    for(auto item : data_mod) {
      llvm::errs() << "Operation " << item.first 
                   << ", occupancy: " << item.second 
                   << ", block number: " << getLogicBB(item.first)
                   << "\n";
    }
    
    ResourceSharing sharing;
    int number_of_basic_blocks = sharing.place_BB(data.archs);

    //perform SCC computation
    llvm::errs() << "\nSCC distribution: ";
    std::vector<int> SCC = sharing.performSCC_bbl();
    int number_of_SCC = SCC.size();
    for(int i = 0; i < number_of_basic_blocks; i++) {
      llvm::errs() << SCC[i] << ", ";
    }
    llvm::errs() << "\n\n";

    TimingDatabase timingDB(&getContext());
    if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
      return signalPassFailure();
    sharing.place_data(data.sharing_feedback, SCC, number_of_SCC, timingDB);
    sharing.print();
    
    //get the Operation we want to add a fork and sink
    mlir::Operation* OUT;
    for(auto item : data_mod) {
      OUT = item.first;
      if(item.first->getName().getStringRef() == "arith.muli") {
        break;
      }
    }
    llvm::errs() << "\nchosen operation: " << OUT << "\n";
    */
    break;
    if (data.someCountOfSomething >= 20) {
      llvm::errs() << "\nBreaking out of the loop!\n";
      break;
    }
  }
}

namespace dynamatic {
namespace sharing {

/// Returns a unique pointer to an operation pass that matches MLIR modules.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createResourceSharingFCCM22Pass(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<ResourceSharingFCCM22Pass>(
      algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs);
}
} // namespace sharing
} // namespace dynamatic