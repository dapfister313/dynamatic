//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Implements the --resource-sharing pass, which checks for sharable
// Operations (sharable means little to no performance overhead).
//===----------------------------------------------------------------------===//

/*
 ****************** What still needs to be implemented ******************
 *
 *  potentially hard:
 *  -
 *
 *  easy:
 *  -
 */

#include "dynamatic/Support/NameUniquer.h"
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

//additional files, remove at the end what not needed
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

namespace dynamatic {
namespace buffer {

  //stores/transfers information needed for resource sharing
  struct ResourceSharingInfo {
    
    // for each CFDFC, store the throughput in double format to 
    // double format to compare 
    std::map<int, double> sharing_check{};
    
    //store stats of each operation
    struct OpSpecific {
      mlir::Operation* op; 
      double occupancy; 
      double op_latency;
      //double throughput;

      void print() {
        llvm::errs() << "Operation " << op 
                    << ", occupancy: " << occupancy 
                    << ", latency: " << op_latency 
                    << ", block: " << getLogicBB(op)
                    << "\n";
      }
    };
    std::vector<OpSpecific> sharing_init;

    //constructor
    ResourceSharingInfo() = default;
  };
 /*
  struct MyFuncInfo : public FuncInfo {
    //vector used to transfer important resource sharing parameters
    ResourceSharingInfo sharing_info;
    std::vector<Value> opaqueChannel;
    MyFuncInfo(): FuncInfo(){};
  };
  */
namespace fpga20 {
  class MyFPGA20Buffers : public FPGA20Buffers {
    //using FPGA20Buffers::FPGA20Buffers;
    public:
    std::vector<ResourceSharingInfo::OpSpecific> getData();
    LogicalResult addSyncConstraints(std::vector<Value> opaqueChannel) {
      for(auto channel : opaqueChannel) {
        ChannelVars &chVars = vars.channels[channel];
        GRBVar &opaque = chVars.bufIsOpaque;
        model.addConstr(opaque == 1.0, "additional_opaque_channel");
      }
      return success();
    }
    //virtual LogicalResult addElasticityConstraints(ValueRange elasticChannels, ArrayRef<Operation *> elasticUnits);
    MyFPGA20Buffers(FuncInfo &funcInfo, const TimingDatabase &timingDB,
                                GRBEnv &env, Logger *logger, double targetPeriod,
                                double maxPeriod, bool legacyPlacement)
        : FPGA20Buffers(funcInfo, timingDB, env, logger, targetPeriod,
                                maxPeriod, legacyPlacement), funcInfo(funcInfo){};
    protected:
    FuncInfo &funcInfo;
  };
  std::vector<ResourceSharingInfo::OpSpecific> MyFPGA20Buffers::getData() {
    std::vector<ResourceSharingInfo::OpSpecific> return_info;
    ResourceSharingInfo::OpSpecific sharing_item;
    double throughput;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcs)) {
      auto [cf, cfVars] = cfdfcWithVars;
      // for each CFDFC, extract the throughput in double format
      throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
      //funcInfo.sharing_info.sharing_check[idx] = throughput;
      
      for (auto &[op, unitVars] : cfVars.units) {
        sharing_item.op = op;
        if (failed(timingDB.getLatency(op, sharing_item.op_latency)) || sharing_item.op_latency == 0.0)
          continue;
        // the occupancy of the unit is calculated as the product between
        // throughput and latency
        sharing_item.occupancy = sharing_item.op_latency * throughput;
        return_info.push_back(sharing_item);
      }
    }
    return return_info;
  }
  /*
  LogicalResult MyFPGA20Buffers::addElasticityConstraints(ValueRange elasticChannels,
                                                        ArrayRef<Operation *> elasticUnits) {
    llvm::errs() << "Elasticity constraint overwritten!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    // Upper bound for the longest rigid path
    unsigned cstCoef = std::distance(funcInfo.funcOp.getOps().begin(),
                                    funcInfo.funcOp.getOps().end()) +
                      2;

    // Add elasticity constraints for channels
    for (Value channel : elasticChannels) {
      ChannelVars &chVars = vars.channels[channel];
      GRBVar &tIn = chVars.tElasIn;
      GRBVar &tOut = chVars.tElasOut;
      GRBVar &present = chVars.bufPresent;
      GRBVar &opaque = chVars.bufIsOpaque;
      GRBVar &numSlots = chVars.bufNumSlots;
      if(std::find(funcInfo.opaqueChannel.begin(), funcInfo.opaqueChannel.end(), channel) != funcInfo.opaqueChannel.end()) {
        model.addConstr(opaque == 1.0, "additional_opaque_channel");
      }
      // If there is an opaque buffer on the channel, the channel elastic
      // arrival time at the ouput must be greater than at the input (breaks
      // cycles!)
      model.addConstr(tOut >= tIn - cstCoef * opaque, "elastic_cycle");
      // If there is an opaque buffer, there must be at least one slot
      model.addConstr(numSlots >= opaque, "elastic_slots");
      // If there is at least one slot, there must be a buffer
      model.addConstr(present >= 0.01 * numSlots, "elastic_present");
    }


    // Add an elasticity constraint for every input/output port pair in the
    // elastic units
    for (Operation *op : elasticUnits) {
      forEachIOPair(op, [&](Value in, Value out) {
        GRBVar &tInPort = vars.channels[in].tElasOut;
        GRBVar &tOutPort = vars.channels[out].tElasIn;
        // The elastic arrival time at the output port must be at least one
        // greater than at the input port
        model.addConstr(tOutPort >= 1 + tInPort, "elastic_unitTime");
      });
    }
    return success();
  }
  */
}
}
}

namespace {

/// Recovered data needed for performing resource sharing
struct ResourceSharing_Data {
  //extracts needed resource sharing data from FuncInfo struct
  ResourceSharingInfo sharing_feedback; 
  //used to perform SCC-computation (finding strongly connected components)
  SmallVector<experimental::ArchBB> archs;

  unsigned someCountOfSomething = 0;
  unsigned totalNumberOfOpaqueBuffers = 0;
  Operation *startingOp;
  std::map<int, dynamatic::sharing::controlStructure> control_map;
  FuncOp funcOp;
  std::vector<Value> opaqueChannel = {};
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
  bool hasCycle;
  
  bool recursivelyDetermineIfCyclic(mlir::Operation* op, std::set<mlir::Operation*>& node_visited, mlir::Operation* ouc) {
    node_visited.insert(op);
    for (auto &u : op->getResults().getUses()) {
      Operation *child_op = u.getOwner();
      if(child_op == ouc) {
        return true;
      }
      auto it = node_visited.find(child_op);
      if(it == node_visited.end()) {
        //not visited yet
        if(recursivelyDetermineIfCyclic(child_op, node_visited, ouc)) {
          return true;
        }
      }
    }
    return false;
  }

  void addOperation(mlir::Operation* op) {
    items.push_back(op);
  }

  bool determineIfCyclic(mlir::Operation* op) {
    std::set<mlir::Operation*> node_visited;
    return recursivelyDetermineIfCyclic(op, node_visited, op);
  }

  //Constructors
  Group(std::vector<mlir::Operation*> ops, double occupancy, bool cyclic)
        : shared_occupancy(occupancy) {
          items = ops;
          hasCycle = cyclic;
        }

  Group(mlir::Operation* op, double occupancy)
        : shared_occupancy(occupancy) {
          items.push_back(op);
          hasCycle = determineIfCyclic(op);
        }
  Group(mlir::Operation* op)
        : shared_occupancy(-1) {
          items.push_back(op);
          hasCycle = determineIfCyclic(op);
        }     

  //Destructor
  ~Group() {};
};

//abbreviation to iterate through list of groups
typedef std::list<Group>::iterator GroupIt;

/*
       Each basic block resides in a set
       of strongly connected components
       For Example: A set could be {1,2,3}
*/
struct Set {
  std::list<Group> groups{};
  int SCC_id;
  double op_latency;
 
  void addGroup(Group group) {
    groups.push_back(group);
  }

  //Constructors
  Set(double latency) {
    op_latency = latency;
  }

  Set(Group group) {
    groups.push_back(group);
  }

  Set(int SCC_idx, double latency) {
    SCC_id = SCC_idx;
    op_latency = latency;
  }

  void joinGroups(GroupIt group1, GroupIt group2, std::vector<mlir::Operation*>& finalOrd) {
    Group newly_created = Group(finalOrd, group1->shared_occupancy + group1->shared_occupancy, group1->hasCycle | group2->hasCycle);
    groups.erase(group1);
    groups.erase(group2);
    groups.push_back(newly_created);
  }

  void joinSet(Set *joined_element) {
    GroupIt pelem = groups.begin();
    for(GroupIt jelem = joined_element->groups.begin(); 
        jelem != joined_element->groups.end(); pelem++, jelem++) {
      pelem->items.insert(pelem->items.end(), 
                          jelem->items.begin(), 
                          jelem->items.end()
                          );
    }
  }
};


/*
 *  Find all combinations of 2 items in a list
 *  Permutations are treated as not unique
 *  Example: group1, group2 is the same as group2, group1
 */
 std::vector<std::pair<GroupIt, GroupIt>> combinations(Set *set) {
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

/*
       Each operation type (e.g. mul, add, load) 
       can be treated separately
*/
struct OpSelector {
  double op_latency;
  llvm::StringRef identifier;
  std::vector<Set> sets{};
  std::map<int, int> SetSelect;
  Set final_grouping;
  std::list<mlir::Operation*> Ops_not_on_CFG;
  
  void addSet(Group group) {
    sets.push_back(Set(group));
  }

  //Constructor
  OpSelector(double latency, llvm::StringRef identifier)
        : op_latency(latency), identifier(identifier), final_grouping(Set(latency)) {
        }
  
  void print() {
    llvm::errs() << identifier << "\n";
    for(auto set : sets) {
      llvm::errs() << "SCC"  << set.SCC_id << ":\n";
      int group_count = 0;
      for(auto group : set.groups) {
        llvm::errs() << "Group " << group_count++ << ": ";
        for(auto item : group.items) {
          llvm::errs() << item << ", ";
        }
      }
      llvm::errs() << "\n";
    }
  }

  void printFinalGroup() {
    llvm::errs() << "Final grouping for " <<identifier << ":\n";
    int group_count = 0;
      for(auto group : final_grouping.groups) {
        llvm::errs() << "Group " << group_count++ << ": ";
        for(auto item : group.items) {
          llvm::errs() << item << ", ";
        }
      }
      llvm::errs() << "\n";
  }

  void sharingAcrossLoopNests() {
    int number_of_sets = sets.size();
    if(!number_of_sets) {
      return;
    }
    
    int max_set_size = -1;
    int max_idx = -1;
    for(int i = 0; i < number_of_sets; i++) {
      if((int)sets[i].groups.size() > max_set_size) {
        max_set_size = sets[i].groups.size();
        max_idx = i;
      }
    }
    //choose initial set 
    final_grouping = sets[max_idx];

    for(int i = 0; i < number_of_sets; i++) {
      if(i == max_idx) {
        continue;
      }
      final_grouping.joinSet(&sets[i]);
    }
  
  }

  void sharingOtherUnits() {
    auto it = final_grouping.groups.begin();
    for(auto unit : Ops_not_on_CFG) {
      it->addOperation(unit);
      it++;
      if(it == final_grouping.groups.end()) {
        it = final_grouping.groups.begin();
      }
    }
  }
};

/*
       Class to iterate easily trough all 
       operation types
*/
class ResourceSharing {
  //troughput per basic block
  std::map<int, double> throughput;
  //connections between basic blocks
  SmallVector<experimental::ArchBB> archs;
  //maps operation types to integers (SCC analysis)
  std::map<llvm::StringRef, int> OpNames;
  //number of sharable operation types
  int number_of_operation_types;
  //operation directly after start
  Operation *firstOp = nullptr;
  //Operations in topological order
  std::map<Operation *, unsigned int> OpTopologicalOrder;
  
  double runPerformanceAnalysis() {
    return 0;
  }
  
  //used to run topological sorting
  void recursiveDFStravel(Operation *op, unsigned int *position, std::set<mlir::Operation*>& node_visited) {
    //add operation
    node_visited.insert(op);
    
    //DFS over all child ops
    for (auto &u : op->getResults().getUses()) {
      Operation *child_op = u.getOwner();
      auto it = node_visited.find(child_op);
      if(it == node_visited.end()) {
        //not visited yet
        recursiveDFStravel(child_op, position, node_visited);
      }
    }
    //update container
    OpTopologicalOrder[op] = *position;
    *position++;
    return;
  }

public:
  std::vector<OpSelector> operation_types;
  
  void setFirstOp(Operation *op) {
    firstOp = op;
  }

  Operation *getFirstOp() {
    return firstOp;
  }

  void initializeTopolocialOpSort() {
    if(firstOp == nullptr) {
      llvm::errs() << "[Error] Operation directly after start not yet present\n";
    }
    unsigned int position = 0;
    std::set<mlir::Operation*> node_visited;
    recursiveDFStravel(firstOp, &position, node_visited);
    return;
  }

  void printTopologicalOrder() {
    llvm::errs() << "Topological Order: \n";
    for(auto [op, id] : OpTopologicalOrder) {
      llvm::errs() << id << " : " << op << "\n";
    }
  }
  
  /*
   * if neighter group 1 nor group 2 are cyclic, we can find a 
   * (not neccesarily unique) topolocical ordering
   */
   std::vector<Operation*> sortTopologically(GroupIt group1, GroupIt group2) {
    std::vector<Operation*> result(group1->items.size() + group2->items.size());
    //add all operations in sorted order
    merge(group1->items.begin(), group1->items.end(), group2->items.begin(), group2->items.end(), result.begin(), [this](Operation *a, Operation *b) {return OpTopologicalOrder[a] > OpTopologicalOrder[b];});
    return result;
   }

   bool isTopologicallySorted(std::vector<Operation*> Ops) {
    for(unsigned long i = 0; i < Ops.size() - 1; i++) {
      if(OpTopologicalOrder[Ops[i]] > OpTopologicalOrder[Ops[i+1]]) {
        return false;
      }
    }
    return true;
   }


  //place resource sharing data retrieved from buffer placement
  void retrieveDataFromPerformanceAnalysis(ResourceSharingInfo sharing_feedback, std::vector<int>& SCC, int number_of_SCC, TimingDatabase timingDB) {
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
        operation_types.push_back(OpSelector(sharing_item.second.second, OpName));
      }
      OpSelector& OpT = operation_types[OpIdx];
      
      //choose the right set
      int SetIdx = -1;
      unsigned int BB = getLogicBB(sharing_item.first).value();
      int SCC_idx = SCC[BB];
      if(SCC_idx == -1) {
        //Operation not part of a set
        OpT.Ops_not_on_CFG.push_back(sharing_item.first);
        continue;
      }
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
  
  //return number of Basic Blocks
  int getNumberOfBasicBlocks() {
    unsigned int maximum = 0;
    for(auto arch_item : archs) {
      maximum = std::max(maximum, std::max(arch_item.srcBB, arch_item.dstBB));
    }
    return maximum + 1; //as we have BB0, we need to add one at the end
  }

  //place retrieved Basic block connections
  void getListOfControlFlowEdges(SmallVector<experimental::ArchBB> archs_ext) {
    archs = archs_ext;
  }
 
  std::vector<int> performSCC_bbl() {
    return Kosarajus_algorithm_BBL(archs);
  }

  void performSCC_opl(std::set<mlir::Operation*>& result) {
    Kosarajus_algorithm_OPL(firstOp, result, OpTopologicalOrder);
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
    for(auto Op : operation_types) {
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
      FuncInfo myInfo = info;
  /// This is exactly the same as the getBufferPlacement method in
  /// HandshakePlaceBuffersPass
  //myInfo.opaqueChannel = data.opaqueChannel;
  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  // Create and solve the MILP
  fpga20::MyFPGA20Buffers *milp = nullptr;
  if (algorithm == "fpga20")
    milp = new fpga20::MyFPGA20Buffers(myInfo, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, false);
  else if (algorithm == "fpga20-legacy")
    milp = new fpga20::MyFPGA20Buffers(myInfo, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, true);
  milp->addSyncConstraints(data.opaqueChannel);
  assert(milp && "unknown placement algorithm");
  int milpStat;
  LogicalResult res = success();
  if (failed(milp->optimize(&milpStat))) {
    res = myInfo.funcOp->emitError()
          << "Buffer placement MILP failed with status " << milpStat
          << ", reason:" << getGurobiOptStatusDesc(milpStat);
  } else if (failed(milp->getPlacement(placement))) {
    res = myInfo.funcOp->emitError()
          << "Failed to extract placement decisions from MILP's solution.";
  }
  data.sharing_feedback.sharing_init = milp->getData();
  
  NameUniquer names(myInfo.funcOp);
  dynamatic::sharing::controlStructure control_item;
  unsigned int BB_idx = 0;
  // Walkin' in the IR:
  llvm::errs() << "Walkin' inside the IR:\n";
  for (Operation &op : myInfo.funcOp.getOps()) {
    if(op.getName().getStringRef() == "handshake.merge" || op.getName().getStringRef() == "handshake.control_merge") {
      for (const auto &u : op.getResults()) {
        if(u.getType().isa<NoneType>()) {
          for(auto &item : u.getUses())
             llvm::errs() << "This is a control channel " << names.getName(op) << " -> " << names.getName(*item.getOwner()) <<"\n";
             BB_idx = getLogicBB(&op).value();
             control_item.control_merge = u;
        }
      }
    }
    if(op.getName().getStringRef() == "handshake.br" || op.getName().getStringRef() == "handshake.cond_br") {
      for (const auto &u : op.getOperands()) {
        if(u.getType().isa<NoneType>()) {
            llvm::errs() << "This is a * control channel " << names.getName(op) << " <- " << names.getName(*u.getDefiningOp()) <<"\n";
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

  // Here, before destroying the MILP, extract whatever information you want
  // and store it into your MyData& reference. If you need to extract variable
  // values from the MILP you may need to make some of its fields public (to
  // be discussed in PRs).
  
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
  data.someCountOfSomething += 10;
  llvm::errs() << "Current count: " << data.someCountOfSomething << "\n";

  //data.sharing_feedback = info.sharing_info;
  data.archs = myInfo.archs;
  data.funcOp = myInfo.funcOp;

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
    for(int i = 0; i < 3; i++) {
      if (failed(pm.run(modOp))) {
        return signalPassFailure();
      }

    }
  
    NameUniquer names(data.funcOp);
    std::unordered_map<mlir::Operation*, double> data_mod;
    for(auto item : data.sharing_feedback.sharing_init) {
      if (data_mod.find(item.op) != data_mod.end()) {
        data_mod[item.op] = std::max(item.occupancy, data_mod[item.op]);
      } else {
        data_mod[item.op] = item.occupancy;
      }
    }
    
    initialize_modification(data.control_map);
    dynamatic::sharing::revert_to_initial_state();

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
                dynamatic::sharing::generate_performance_model(&builder, current_permutation);
                dynamatic::sharing::deleteAllBuffers(data.funcOp);
                if (failed(pm.run(modOp))) {
                  return signalPassFailure();
                }
                dynamatic::sharing::destroy_performance_model(&builder, current_permutation);
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
        return_values.push_back(dynamatic::sharing::generate_performance_step(&builder, op));
      }

    data.opaqueChannel = return_values;
    dynamatic::sharing::deleteAllBuffers(data.funcOp);
    if (failed(pm.run(modOp))) {
      return signalPassFailure();
    }
    //dynamatic::sharing::revert_performance_step(&builder, op);
    break;
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