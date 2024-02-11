#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGFRAMEWORK_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGFRAMEWORK_H

#include "experimental/Transforms/ResourceSharing/FCCM22Sharing.h"
#include "experimental/Transforms/ResourceSharing/SCC.h"
#include "experimental/Transforms/ResourceSharing/modIR.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
//#include "experimental/Transforms/ResourceSharing/SharingSupport.h"
#include "mlir/Pass/PassManager.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"

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
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::sharing;

//std::max
#include <algorithm>
#include <list>

namespace dynamatic {
namespace experimental {
namespace sharing {

//stores/transfers information needed for resource sharing from buffer placement
struct ResourceSharingInfo {
    // for each CFDFC, store the throughput in double format to double format to compare
    std::map<int, double> throughputPerCFDFC{};

    // stores shareable operations and their occupancy
    struct OperationData {
        mlir::Operation* op;
        double occupancy;

        void print();
    };
    std::vector<OperationData> operations;
    
    // used to perform SCC-computation (finding strongly connected components)
    SmallVector<dynamatic::experimental::ArchBB> archs;

    // handshake::FuncOp
    FuncOp funcOp;

    // list of values where to insert a seq buffer
    std::vector<Value> opaqueChannel = {};
    
    // determines if one should give the full report back or just the current occupancy sum
    bool fullReportRequired = true;
    
    // specific cluster of operations
    std::vector<Operation*> testedGroups;
    // occupancy sum of specific cluster of operations (see above)
    double occupancySum;
    void computeOccupancySum();

    //constructor
    ResourceSharingInfo() = default;
};

/*
       Inside each set of strongly connected components
       one needs to check if sharing is possible
*/
struct Group {
  std::vector<mlir::Operation*> items;
  double shared_occupancy;
  bool hasCycle;
  
  // determine if an operation is cyclic (if there is a path from the op that reaches back to it)
  bool recursivelyDetermineIfCyclic(mlir::Operation* current_op, std::set<mlir::Operation*>& node_visited, mlir::Operation* op);
  bool determineIfCyclic(mlir::Operation* op);

  // add operation to group
  // important: changes neither the shared occupancy nor the hasCycle attributes
  void addOperation(mlir::Operation* op);

  // Constructors
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
  
  // add group to set
  void addGroup(Group group);
  
  // merge two existing groups (of this specific set) into one newly created group
  void joinGroups(GroupIt group1, GroupIt group2, std::vector<mlir::Operation*>& finalOrd);
  
  // join another set to this specific set 
  // important: while joining, one group of each set is paired with one group of the other set
  void joinSet(Set *joined_element);

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
};

/*
       Each operation type (e.g. mul, add, load)
       can be treated separately
*/
//OperationTypeStruct
struct OpSelector {
  double op_latency;
  llvm::StringRef identifier;
  std::vector<Set> sets{};
  std::map<int, int> SetSelect;
  Set final_grouping;
  std::list<mlir::Operation*> Ops_not_on_CFG;
  
  // add set to operation type
  void addSet(Group group);
  
  // print the composition of Sets/SCCs - Groups
  void print();
  
  // the end of the sharing strategy joins sets together, use this to print the final set
  void printFinalGroup();
  
  // joines the sets at the end of the sharing algorithm
  void sharingAcrossLoopNests();
  
  // joines operations that do not reside in a CFDFC
  void sharingOtherUnits();

  //Constructor
  OpSelector(double latency, llvm::StringRef identifier)
        : op_latency(latency), identifier(identifier), final_grouping(Set(latency)) {}

};

/*
       Class to iterate easily trough all
       operation types
*/
class ResourceSharing {
  // troughput per CFDFC
  std::map<int, double> throughput;
  // connections between basic blocks
  SmallVector<dynamatic::experimental::ArchBB> archs;
  // maps operation types to integers (SCC analysis)
  std::map<llvm::StringRef, int> OpNames;
  // number of sharable operation types
  int number_of_operation_types;
  // operation directly after start
  Operation *firstOp = nullptr;
  // Operations in topological order
  std::map<Operation *, unsigned int> OpTopologicalOrder;
  
  // run performance ananlysis here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  double runPerformanceAnalysis();

  // used to run topological sorting
  void recursiveDFStravel(Operation *op, unsigned int *position, std::set<mlir::Operation*>& node_visited);

public:
  // stores control merge and branch of each BB
  std::map<int, controlStructure> control_map;

  std::vector<OpSelector> operation_types;
  
  // set first operation of the IR
  void setFirstOp(Operation *op);

  // get first operation of the IR
  Operation *getFirstOp();

  // compute first operation of the IR
  bool computeFirstOp(FuncOp funcOp);
  
  // calculate the topological ordering of all operations
  // important: operations on a cycle do not have a topological order
  //            but are still present
  void initializeTopolocialOpSort();

  // print operations in topological order
  void printTopologicalOrder();
  
  // sort operations in two groups topologically
  std::vector<Operation*> sortTopologically(GroupIt group1, GroupIt group2);
  
  // determine if a vector of operations are in topological order
  bool isTopologicallySorted(std::vector<Operation*> Ops);

  // place resource sharing data retrieved from buffer placement
  void retrieveDataFromPerformanceAnalysis(ResourceSharingInfo sharing_feedback, std::vector<int>& SCC, int number_of_SCC, TimingDatabase timingDB);

  // return number of Basic Blocks
  int getNumberOfBasicBlocks();

  // place retrieved connections between Basic blocks
  void getListOfControlFlowEdges(SmallVector<dynamatic::experimental::ArchBB> archs_ext);
  
  // perform SCC-agorithm on basic block level
  std::vector<int> performSCC_bbl();
  
  // perform SCC-agorithm on operation level
  void performSCC_opl(std::set<mlir::Operation*>& result);
  
  // print source-destination BB of connection between BBs, throughput per CFDFC and 
  // the composition in operation-type, set, group
  void print();
  
  // find control structure of each BB: control_merge, control_branch
  void getControlStructure(FuncOp funcOp);

  // place and compute all necessary data to perform resource sharing
  void placeAndComputeNecessaryDataFromPerformanceAnalysis(ResourceSharingInfo data, TimingDatabase timingDB);

  // constructor
  ResourceSharing(ResourceSharingInfo data, TimingDatabase timingDB) {
     placeAndComputeNecessaryDataFromPerformanceAnalysis(data, timingDB);
  }
};

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif //EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGFRAMEWORK_H