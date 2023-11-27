//===- HandshakeIterativeBuffers.cpp - Iter. buffer placement ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <TODO What does this file implement?>
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <list>
#include <map>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;


namespace {
/*
struct OpIdentifier {
  std::map<llvm::StringRef, mlir::Operation*> bridge{};
  OpIdentifier(ModuleOp modOp) {
    for (mlir::Region &region : modOp->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        block.walk([&](mlir::Operation *op) {
          bridge[op->getName().getStringRef()] = op;
        });
      }
    }
  }
};

struct OpItem {
  mlir::Operation* op; 
  std::string op_str; 
  double occupancy; 
  double op_latency;
  double throughput;
  int cfdfc_id;
  
  bool get_op(OpIdentifier matching) { 
    auto id = matching.bridge.find(op_str);
    if(id != matching.bridge.end()) {
      op = id->second;
    }
    return 0;
  }

  void print_OpItem() {
    llvm::errs() << "Op:" << op_str << "\n";
    llvm::errs() << "occupancy:" << occupancy << "\n";
    llvm::errs() << "op_latency:" << op_latency << "\n";
    llvm::errs() << "throughput:" << throughput << "\n";
    llvm::errs() << "cfdfc_id:" << cfdfc_id << "\n";
  }
};



struct Group {
  std::vector<mlir::Operation*> items;
  double shared_occupancy;
  int cfdfc_id;

  //Constructors
  Group(mlir::Operation* op, double occupancy, int cfdfc_id)
        : shared_occupancy(occupancy), cfdfc_id(cfdfc_id) {
          items.push_back(op);
        }
  Group(OpItem item)
        : shared_occupancy(item.occupancy),  cfdfc_id(item.cfdfc_id) {
          items.push_back(item.op);
        }


  //Destructor
  ~Group() {};
};

struct Set {
  std::list<Group> groups{};
  double op_latency;
  llvm::StringRef identifier;
  bool modified;

  Set(OpItem item)
        : op_latency(item.op_latency), identifier(item.op->getName().getStringRef()),
          modified(true) {
            groups.push_front(Group(item));
          }
  
  int try_grouping(Group *elem1, Group *elem2) {
    if(elem1->cfdfc_id == elem2->cfdfc_id) {
      //belong to the same cfdfc loop
      if(elem1->shared_occupancy + elem2->shared_occupancy <= op_latency) {
        //join groups
      }
    }
  }

  int join_groups(std::list<Group> groups) {
    for(std::list<Group>::iterator outer_it = groups.begin(); outer_it != groups.end(); ++outer_it) {
      for(std::list<Group>::iterator inner_it = outer_it + 1; inner_it != groups.end(); ++inner_it) {

      }
    }
  }
  
};

class ResourceSharing {
  std::vector<Set> sets{};
  std::map<llvm::StringRef, int> set_selector;
  int number_of_sets;

  int get_id(mlir::Operation *op) {
    auto id = set_selector.find(op->getName().getStringRef());
    if(id == set_selector.end()) {
      return -1;
    }
    return id->second;
  }

  void set_id(mlir::Operation *op) {
    set_selector[op->getName().getStringRef()] = number_of_sets;
  }

public:
  
  void add_operation(OpItem item) {
    int id = get_id(item.op);
    if(id != -1) {
      //set already exists
      sets[id].groups.push_front(Group(item));
    } else {
      //create set
      sets.push_back(Set(item));
      set_id(item.op);
    }
  }

  void print_setup() {
    for(auto set : sets) {
      llvm::errs() << "Placeholder\n";
    }
  }
  
};
*/

/*
       Inside each set of strongly connected components
       one needs to check if sharing is possible
*/
struct Group {
  std::vector<mlir::Operation*> items;
  double shared_occupancy;
  double InitThroughput;

  //Constructors
  Group(mlir::Operation* op, double occupancy, double throughput)
        : shared_occupancy(occupancy), InitThroughput(throughput) {
          items.push_back(op);
        }
  Group(mlir::Operation* op)
        : shared_occupancy(-1), InitThroughput(-1) {
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
  
  void addGroup(Group group) {
    groups.push_back(group);
  }

  //Constructor
  Set(Group group) {
    groups.push_back(group);
  }
};

/*
       Each operation type (e.g. mul, add, load) 
       can be treated separately
*/
struct OpSelector {
  double op_latency;
  llvm::StringRef identifier;
  std::vector<Set> sets{};
  
  void addSet(Group group) {
    sets.push_back(Set(group));
  }

  //Constructor
  OpSelector(double latency, llvm::StringRef identifier, Group entry)
        : op_latency(latency), identifier(identifier) {
          sets.push_back(Set(entry));
        }

};

/*
       Class to iterate easily trough all 
       operation types
*/
class ResourceSharing {
  std::vector<OpSelector> operation_type{};
  //std::map<llvm::StringRef, int> op_selector; //could be used to eaily determine the operation if there are a lot
  
  /*
  void addOpType(ResourceSharingInfo RSitem, llvm::StringRef OpName) {
    operation_type
  }
  */
  double runPerformanceAnalysis() {
    return 0;
  }

public:
  //place resource sharing data retrieved from buffer placement
  void place_data(std::vector<ResourceSharingInfo> sharing_feedback) {
    for(auto sharing_item : sharing_feedback) {
      llvm::StringRef OpName = sharing_item.op->getName().getStringRef();
      Group item = Group(sharing_item.op, sharing_item.occupancy, sharing_item.throughput);
      int loc = -1;
      for(unsigned long i = 0; i < operation_type.size(); i++) {
        if(OpName == operation_type[i].identifier) {
          loc = i;
          break;
        }
      }
      
      if(loc == -1) {
        operation_type.push_back(OpSelector(sharing_item.op_latency, OpName, item));
      } else {
        operation_type[loc].sets[0].groups.push_front(item);
      }
    }
  }

  void print() {
    for(auto Op : operation_type) {
      llvm::errs() << "*** New Operation type: " << Op.identifier << " ***\n";
      for(auto set : Op.sets) {
        llvm::errs() << "** New set **\n";
        for(auto group : set.groups) {
          llvm::errs() << "* New group *\n";
          llvm::errs() << "Throughput: " << group.InitThroughput << "\n";
          llvm::errs() << "Number of entries: " << group.items.size() << "\n";
        }
      }
    }
  }
};

/// Stores some data you may want to extract from buffer placement
struct MyData {
  //extracts needed resource sharing data from FuncInfo struct
  std::vector<ResourceSharingInfo> sharing_feedback; 
  unsigned someCountOfSomething = 0;
  unsigned totalNumberOfOpaqueBuffers = 0;
};

/// Sub-type of the classic buffer placement pass, just so that we can override
/// some of the methods used during buffer placement and extract internal data
/// (e.g., channel throughputs as determined by the MILP) from the pass.
struct MyBufferPlacementPass : public HandshakePlaceBuffersPass {
  MyBufferPlacementPass(MyData &data, StringRef algorithm,
                        StringRef frequencies, StringRef timingModels,
                        bool firstCFDFC, double targetCP, unsigned timeout,
                        bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        data(data){};

  /// Some data you care about extracting.
  MyData &data;

protected:
  /// Custom buffer placement step. Copied from `HandshakePlaceBuffersPass` with
  /// the addition of a step at the end to extract some information from the
  /// MILP.
  LogicalResult
  getBufferPlacement(FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
                     DenseMap<Value, PlacementResult> &placement) override;
};
} // namespace

LogicalResult MyBufferPlacementPass::getBufferPlacement(
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
  llvm::errs() << "Walkin' inside the IR:\n";
  for (Operation &op : info.funcOp.getOps()) {
    llvm::errs() << "Operation" << op << "\n";
  }

  // Here, before destroying the MILP, extract whatever information you want
  // and store it into your MyData& reference. If you need to extract variable
  // values from the MILP you may need to make some of its fields public (to
  // be discussed in PRs).
  llvm::errs() << "Setting some random count!\n";
  data.someCountOfSomething += 10;
  llvm::errs() << "Current count: " << data.someCountOfSomething << "\n";

  data.sharing_feedback = info.sharing_info;
  data.someCountOfSomething += 10;
  delete milp;
  return res;
}

namespace {
/// Driver for the iterative buffer placement pass, which runs buffer placement
/// and something else of your choosing iteratively until some exit condition is
/// met.
struct HandshakeIterativeBuffersPass
    : public dynamatic::buffer::impl::HandshakeIterativeBuffersBase<
          HandshakeIterativeBuffersPass> {

  /// Note that I made the pass take exactly the same set of parameters as the
  /// buffer placement pass for completeness. If some of them are useless to you
  /// (e.g., you may only want to ever use the 'fpga20' algorithm, you can
  /// remove them from there and from everywhere they are mentionned)
  HandshakeIterativeBuffersPass(StringRef algorithm, StringRef frequencies,
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

/*
std::vector<std::vector<Op_stats>> approx_join(std::vector<Op_stats> information) {
  std::vector<std::vector<Op_stats>> result;
  std::vector<double> capacity;
  bool taken;
  for(auto opT : information) {
    taken = 0;
    double temp = 1/opT.occupancy;
    for(unsigned long i = 0; i < capacity.size(); ++i) {
      if(capacity[i] + temp <= 1) {
        capacity[i] += temp;
        result[i].push_back(opT);
        taken = 1;
        break;
      }
    }
    if(!taken) {
       int lat = result.size();
       result.resize(lat+1);
       capacity.push_back(temp);
       result[lat].push_back(opT);
    }
  }
  return result;
}


std::vector<std::vector<Op_stats>> recursive_join(
            std::vector<Op_stats> information, 
            std::vector<std::vector<Op_stats>> result,
            int max_slots, int depth, int max_depth) {
    if(depth == max_depth) {
      return result;
    }

    return result;
}

std::vector<std::vector<Op_stats>> best_join(std::vector<Op_stats> information) {
  double comp_max_slots = 0;
  for(auto opT : information) {
    comp_max_slots += 1/opT.occupancy;
  }
  int max_slots = 2*ceil(comp_max_slots);
  std::vector<std::vector<Op_stats>> result(max_slots);
  std::vector<double> capacity;
  return result;
}
*/


void HandshakeIterativeBuffersPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  MyData data;

  while (true) {
    // Data object to extract information from buffer placement
    // Use a pass manager to run buffer placement on the current module
    PassManager pm(&getContext());
    pm.addPass(std::make_unique<MyBufferPlacementPass>(
        data, algorithm, frequencies, timingModels, firstCFDFC, targetCP,
        timeout, dumpLogs));
    if (failed(pm.run(modOp))) {
      return signalPassFailure();
    }

    // At this point modOp is buffered. Now you can:
    // - further modify the module by applying any kind of transformation you
    //   want
    // - break out of the loop
    // - ...$
    llvm::errs() << "Hi!\n";
    for(auto item : data.sharing_feedback) {
      item.print();
    }

    ResourceSharing sharing;
    sharing.place_data(data.sharing_feedback);
    sharing.print();

    if (data.someCountOfSomething >= 20) {
      llvm::errs() << "Breaking out of the loop!\n";
      break;
    }
    
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakeIterativeBuffers(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<HandshakeIterativeBuffersPass>(
      algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs);
}
