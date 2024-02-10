#include "experimental/Transforms/ResourceSharing/SharingFramework.h"

using namespace dynamatic::experimental::sharing;

void ResourceSharingInfo::OperationData::print() {
    llvm::errs() << "Operation " << op
                << ", occupancy: " << occupancy
                << ", latency: " << op_latency
                << ", block: " << getLogicBB(op)
                << "\n";
}

void Group::addOperation(mlir::Operation* op) {
    items.push_back(op);
}

bool Group::recursivelyDetermineIfCyclic(mlir::Operation* op, std::set<mlir::Operation*>& node_visited, mlir::Operation* ouc) {
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

void OpSelector::addSet(Group group) {
    sets.push_back(Set(group));
}

void OpSelector::print() {
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

void OpSelector::printFinalGroup() {
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

void OpSelector::sharingAcrossLoopNests() {
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

void OpSelector::sharingOtherUnits() {
    auto it = final_grouping.groups.begin();
    for(auto unit : Ops_not_on_CFG) {
        it->addOperation(unit);
        it++;
        if(it == final_grouping.groups.end()) {
            it = final_grouping.groups.begin();
        }
    }
}

double ResourceSharing::runPerformanceAnalysis() {
    return 0;
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
    *position++;
    return;
}

void ResourceSharing::setFirstOp(Operation *op) {
    firstOp = op;
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
        if (failed(timingDB.getLatency(sharing_item.first, SignalType::DATA, latency)))
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
