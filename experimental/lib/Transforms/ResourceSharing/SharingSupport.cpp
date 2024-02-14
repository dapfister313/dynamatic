#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

//using namespace dynamatic::experimental::sharing;

using namespace dynamatic::buffer::fpga20;


/*
 * Extending FPGA20Buffers class
 */

std::vector<ResourceSharingInfo::OperationData> MyFPGA20Buffers::getData() {
    std::vector<ResourceSharingInfo::OperationData> return_info;
    ResourceSharingInfo::OperationData sharing_item;
    double throughput, latency;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
        auto [cf, cfVars] = cfdfcWithVars;
        throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

        for (auto &[op, unitVars] : cfVars.unitVars) {
        sharing_item.op = op;
        if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) || latency == 0.0)
            continue;
        sharing_item.occupancy = latency * throughput;
        return_info.push_back(sharing_item);
        }
    }
    return return_info;
}

double MyFPGA20Buffers::getOccupancySum(std::set<Operation*>& group) {
    std::map<Operation*, double> occupancies;
    for(auto item : group) {
        occupancies[item] = -1.0;
    }
    double throughput, latency, occupancy;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
        auto [cf, cfVars] = cfdfcWithVars;
        // for each CFDFC, extract the throughput in double format
        throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

        for (auto &[op, unitVars] : cfVars.unitVars) {
            if(group.find(op) != group.end()) {
                if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) || latency == 0.0)
                    continue;
                occupancy = latency * throughput;
                occupancies[op] = std::max(occupancy, occupancies[op]);
            }
        }
    }
    double sum = 0.0;
    for(auto item : occupancies) {
        assert(item.second > 0 && "Incorrect occupancy\n");
        sum += item.second;
    }
    return sum;
}

LogicalResult MyFPGA20Buffers::addSyncConstraints(std::vector<Value> opaqueChannel) {
    for(auto channel : opaqueChannel) {
        ChannelVars &chVars = vars.channelVars[channel];
        auto dataIt = chVars.signalVars.find(SignalType::DATA);
        GRBVar &dataOpaque = dataIt->second.bufPresent;
        GRBVar &opaque = chVars.bufPresent;
        model.addConstr(opaque == 1.0, "additional_opaque_channel");
        model.addConstr(dataOpaque == 1.0, "additional_opaque_channel");
    }
    return success();
}


/*
 * additional functions used for resource sharing
 */

bool dynamatic::experimental::sharing::lessOrEqual(double a, double b) {
    double diff = 0.000001;
    if((a < b + diff)) {
        return true;
    }
    return false;
}

bool dynamatic::experimental::sharing::equal(double a, double b) {
    double diff = 0.000001;
    if((a + diff > b)  && (b + diff > a)) {
        return true;
    }
    return false;
}

std::vector<std::pair<GroupIt, GroupIt>> dynamatic::experimental::sharing::combinations(Set *set) {
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
 *   indroducing version of next_permutation
 */

void permutation::findBBEdges(std::deque<std::pair<int, int>>& BBops, std::vector<Operation*>& permutation_vector) {
    std::sort(permutation_vector.begin(), permutation_vector.end(), [](Operation *a, Operation *b) -> bool {return (getLogicBB(a) < getLogicBB(b)) || (getLogicBB(a) == getLogicBB(b) && (a < b));});
    int size = permutation_vector.size();
    int start, end = 0;
    while(end != size) {
        start = end;
        unsigned int BasicBlockId = getLogicBB(permutation_vector[start]).value();
        while(end != size && getLogicBB(permutation_vector[end]).value() == BasicBlockId) {
            ++end;
        }
        BBops.push_front(std::make_pair(start, end));
    }
}

bool permutation::get_next_permutation(PermutationEdge begin_of_permutation_vector, std::deque<std::pair<int, int>>& separation_of_BBs) {
    for(auto [start, end] : separation_of_BBs) {
        if(next_permutation (begin_of_permutation_vector + start, begin_of_permutation_vector + end)) {
          return true;
        }
    }
    return false;
}

