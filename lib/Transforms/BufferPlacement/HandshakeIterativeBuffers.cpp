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
#include <unordered_set>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;


namespace {

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
  std::vector<OpSelector> operation_type{};
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
      double latency;
      if (failed(timingDB.getLatency(sharing_item.first, latency)))
        latency = 0.0;
      llvm::errs() << "Latency of unit " << sharing_item.first << ": " << latency << "\n";
      llvm::StringRef OpName = sharing_item.first->getName().getStringRef();
      Group group_item = Group(sharing_item.first, sharing_item.second.first);
      unsigned int BB = getLogicBB(sharing_item.first).value();
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
      operation_type[BB].sets;
    }
    throughput = sharing_feedback.sharing_check;
  }

  void place_BB(SmallVector<experimental::ArchBB> archs_ext) {
    archs = archs_ext;
  }
  
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
  data.archs = info.archs;
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
    llvm::errs() << "Initally:\n";
    
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
                   << "block number: " << getLogicBB(item.first)
                   << "\n";
    }
    
    ResourceSharing sharing;
    sharing.place_BB(data.archs);

    //finding strongly connected components
    int number_of_basic_blocks = 7;
    std::vector<int> SCC(number_of_basic_blocks);
    int number_of_SCC = sharing.find_strongly_connected_components(SCC);
    llvm::errs() << "Number of strongly connected components: " << number_of_SCC << "\n";
    for(int i = 0; i < number_of_basic_blocks; i++) {
      llvm::errs() << SCC[i] << ", ";
    }
    llvm::errs() << "\n";
    TimingDatabase timingDB(&getContext());
    if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
      return signalPassFailure();
    //sharing.place_data(data.sharing_feedback, SCC, number_of_SCC, timingDB);
    sharing.print();
    
    if (data.someCountOfSomething >= 20) {
      llvm::errs() << "\nBreaking out of the loop!\n";
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
