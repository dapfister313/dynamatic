#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGFRAMEWORK_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGFRAMEWORK_H

//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Implements the --resource-sharing pass, which checks for sharable
// Operations (sharable means little to no performance overhead).
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/TimingModels.h"
#include "experimental/Transforms/ResourceSharing/SCC.h"
#include "experimental/Transforms/ResourceSharing/modIR.h"

using namespace dynamatic;
using namespace dynamatic::handshake;

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
    std::set<Operation*> testedGroups;
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
  
  // print content of specific set
  void print();

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
struct ResourceSharingForSingleType {
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
  ResourceSharingForSingleType(double latency, llvm::StringRef identifier)
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

  // used to run topological sorting
  void recursiveDFStravel(Operation *op, unsigned int *position, std::set<mlir::Operation*>& node_visited);

public:

  // stores control merge and branch of each BB
  std::map<int, controlStructure> control_map;

  std::vector<ResourceSharingForSingleType> operationTypes;
  
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