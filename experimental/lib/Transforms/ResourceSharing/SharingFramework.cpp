#include "experimental/Transforms/ResourceSharing/SharingFramework.h"

using namespace dynamatic::experimental::sharing;

void ResourceSharingInfo::OperationData::print() {
    llvm::errs() << "Operation " << op
                << ", occupancy: " << occupancy
                << ", block: " << getLogicBB(op)
                << "\n";
}

void ResourceSharingInfo::computeOccupancySum() {
    return;
}

void Group::addOperation(mlir::Operation* op) {
    items.push_back(op);
}

bool Group::recursivelyDetermineIfCyclic(mlir::Operation* current_op, std::set<mlir::Operation*>& node_visited, mlir::Operation* op) {
    node_visited.insert(current_op);
    for (auto &u : current_op->getResults().getUses()) {
        Operation *child_op = u.getOwner();
        if(child_op == op) {
            return true;
        }
        auto it = node_visited.find(child_op);
        if(it == node_visited.end()) {
            //not visited yet
            if(recursivelyDetermineIfCyclic(child_op, node_visited, op)) {
                return true;
            }
        }
    }
    return false;
}

bool Group::determineIfCyclic(mlir::Operation* op) {
    std::set<mlir::Operation*> node_visited;
    return recursivelyDetermineIfCyclic(op, node_visited, op);
}

void Set::addGroup(Group group) {
    groups.push_back(group);
}

void Set::joinGroups(GroupIt group1, GroupIt group2, std::vector<mlir::Operation*>& finalOrd) {
    Group newly_created = Group(finalOrd, group1->shared_occupancy + group1->shared_occupancy, group1->hasCycle | group2->hasCycle);
    groups.erase(group1);
    groups.erase(group2);
    groups.push_back(newly_created);
}

void Set::joinSet(Set *joined_element) {
    GroupIt pelem = groups.begin();
    for(GroupIt jelem = joined_element->groups.begin(); jelem != joined_element->groups.end(); pelem++, jelem++) {
        pelem->items.insert(pelem->items.end(),
                            jelem->items.begin(),
                            jelem->items.end()
                            );
    }
}
// std::vector<mlir::Operation*> items;
void Set::print(NameUniquer names) {
    llvm::errs() << "Set id: " << SCC_id << "\n";
    int i = 0;
    for(auto group : groups) {
        llvm::errs() << "Group #" << i++ << "\n";
        for(auto op : group.items) {
            llvm::errs() << names.getName(*op) << ", ";
        }
        llvm::errs() << "\n";
    }
}

void ResourceSharingForSingleType::addSet(Group group) {
    sets.push_back(Set(group));
}

void ResourceSharingForSingleType::print() {
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

void ResourceSharingForSingleType::printFinalGroup() {
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

void ResourceSharingForSingleType::sharingAcrossLoopNests() {
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

void ResourceSharingForSingleType::sharingOtherUnits() {
    auto it = final_grouping.groups.begin();
    for(auto unit : Ops_not_on_CFG) {
        it->addOperation(unit);
        it++;
        if(it == final_grouping.groups.end()) {
            it = final_grouping.groups.begin();
        }
    }
}

void ResourceSharing::recursiveDFStravel(Operation *op, unsigned int *position, std::set<mlir::Operation*>& node_visited) {
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
    ++(*position);
    return;
}

bool ResourceSharing::computeFirstOp(FuncOp funcOp) {
    // If we are in the entry block, we can use the start input of the
    // function (last argument) as our control value
    if(!funcOp.getArguments().back().getType().isa<NoneType>()) {
        return false;
    }
    Value func = funcOp.getArguments().back();
    std::vector<Operation *> startingOps;
    for (auto &u : func.getUses())
        startingOps.push_back(u.getOwner());
    if(startingOps.size() != 1) {
        return false;
    }
    firstOp = startingOps[0];
    return true;
}

Operation *ResourceSharing::getFirstOp() {
    return firstOp;
}

void ResourceSharing::initializeTopolocialOpSort() {
    if(firstOp == nullptr) {
        llvm::errs() << "[Error] Operation directly after start not yet present\n";
    }
    unsigned int position = 0;
    std::set<mlir::Operation*> node_visited;
    recursiveDFStravel(firstOp, &position, node_visited);
    return;
}

void ResourceSharing::printTopologicalOrder() {
    llvm::errs() << "Topological Order: \n";
    for(auto [op, id] : OpTopologicalOrder) {
        llvm::errs() << id << " : " << op << "\n";
    }
}

std::vector<Operation*> ResourceSharing::sortTopologically(GroupIt group1, GroupIt group2) {
    std::vector<Operation*> result(group1->items.size() + group2->items.size());
    //add all operations in sorted order
    merge(group1->items.begin(), group1->items.end(), group2->items.begin(), group2->items.end(), result.begin(), [this](Operation *a, Operation *b) {return OpTopologicalOrder[a] > OpTopologicalOrder[b];});
    return result;
}

bool ResourceSharing::isTopologicallySorted(std::vector<Operation*> Ops) {
    for(unsigned long i = 0; i < Ops.size() - 1; i++) {
        if(OpTopologicalOrder[Ops[i]] > OpTopologicalOrder[Ops[i+1]]) {
            return false;
        }
    }
    return true;
}

void ResourceSharing::retrieveDataFromPerformanceAnalysis(ResourceSharingInfo sharing_feedback, std::vector<int>& SCC, int number_of_SCC, TimingDatabase timingDB) {
    // take biggest occupancy per operation
    std::unordered_map<mlir::Operation*, double> uniqueOperation;
    for(auto item : sharing_feedback.operations) {
        if (uniqueOperation.find(item.op) != uniqueOperation.end()) {
            // operation already present
            uniqueOperation[item.op] = std::max(item.occupancy, uniqueOperation[item.op]);
        } else {
            // add operation
            uniqueOperation[item.op] = item.occupancy;
        }
    }

    //everytime we place/overwrite data, initial number of operation types is 0;
    number_of_operation_types = 0;

    //iterate through all retrieved operations
    for(auto op : uniqueOperation) {
        //choose the right operation type
        double latency;
        if (failed(timingDB.getLatency(op.first, SignalType::DATA, latency)))
            latency = 0.0;

        llvm::StringRef OpName = op.first->getName().getStringRef();
        Group group_item = Group(op.first, op.second);
        int OpIdx = -1;
        auto item = OpNames.find(OpName);
        if(item != OpNames.end()) {
            OpIdx = item->second;
        } else {
            OpNames[OpName] = number_of_operation_types;
            OpIdx = number_of_operation_types;
            ++number_of_operation_types;
            operation_types.push_back(ResourceSharingForSingleType(latency, OpName));
        }
        ResourceSharingForSingleType& OpT = operation_types[OpIdx];

        //choose the right set
        int SetIdx = -1;
        unsigned int BB = getLogicBB(op.first).value();
        int SCC_idx = SCC[BB];
        if(SCC_idx == -1) {
            //Operation not part of a set
            OpT.Ops_not_on_CFG.push_back(op.first);
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
}

int ResourceSharing::getNumberOfBasicBlocks() {
    unsigned int maximum = 0;
    for(auto arch_item : archs) {
        maximum = std::max(maximum, std::max(arch_item.srcBB, arch_item.dstBB));
    }
    return maximum + 1; //as we have BB0, we need to add one at the end
}

void ResourceSharing::getListOfControlFlowEdges(SmallVector<dynamatic::experimental::ArchBB> archs_ext) {
    archs = archs_ext;
}

std::vector<int> ResourceSharing::performSCC_bbl() {
    return Kosarajus_algorithm_BBL(archs);
}

void ResourceSharing::performSCC_opl(std::set<mlir::Operation*>& result) {
    Kosarajus_algorithm_OPL(firstOp, result, OpTopologicalOrder);
}

void ResourceSharing::print() {
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

void ResourceSharing::getControlStructure(FuncOp funcOp) {
    controlStructure control_item;
    unsigned int BB_idx = 0;
    for (Operation &op : funcOp.getOps()) {
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
                    control_map[BB_idx] = control_item;
                }
            }
        }
    }
    return;
}

void ResourceSharing::placeAndComputeNecessaryDataFromPerformanceAnalysis(ResourceSharingInfo data, TimingDatabase timingDB) {
    // comput first operation of the IR
    computeFirstOp(data.funcOp);

    // initialize topological sorting to determine topological order
    initializeTopolocialOpSort();
    
    // find non-cyclic operations
    std::set<mlir::Operation*> ops_with_no_loops;
    performSCC_opl(ops_with_no_loops);
    
    // get the connections between basic blocks
    getListOfControlFlowEdges(data.archs);

    // perform SCC computation
    std::vector<int> SCC = performSCC_bbl();

    // get number of strongly connected components
    int number_of_SCC = SCC.size();
    
    // fill resource sharing class with all shareable operations
    retrieveDataFromPerformanceAnalysis(data, SCC, number_of_SCC, timingDB);
    
    // get control structures of each BB
    getControlStructure(data.funcOp);
}
