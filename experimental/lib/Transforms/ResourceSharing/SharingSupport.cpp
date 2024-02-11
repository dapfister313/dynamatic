#include "experimental/Transforms/ResourceSharing/SharingSupport.h"

using namespace dynamatic::experimental::sharing;

using namespace dynamatic::buffer::fpga20;

std::vector<ResourceSharingInfo::OperationData> MyFPGA20Buffers::getData() {
    std::vector<ResourceSharingInfo::OperationData> return_info;
    ResourceSharingInfo::OperationData sharing_item;
    double throughput, latency;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
        auto [cf, cfVars] = cfdfcWithVars;
        // for each CFDFC, extract the throughput in double format
        throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

        for (auto &[op, unitVars] : cfVars.unitVars) {
        sharing_item.op = op;
        if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) || latency == 0.0)
            continue;
        // the occupancy of the unit is calculated as the product between
        // throughput and latency
        sharing_item.occupancy = latency * throughput;
        return_info.push_back(sharing_item);
        }
    }
    return return_info;
}

LogicalResult MyFPGA20Buffers::addSyncConstraints(std::vector<Value> opaqueChannel) {
    for(auto channel : opaqueChannel) {
        ChannelVars &chVars = vars.channelVars[channel];
        GRBVar &opaque = chVars.bufPresent;
        model.addConstr(opaque == 1.0, "additional_opaque_channel");
    }
    return success();
}