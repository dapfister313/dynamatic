#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H

#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "experimental/Transforms/ResourceSharing/SharingFramework.h"

using namespace dynamatic::experimental::sharing;

namespace dynamatic {
namespace experimental {
namespace sharing {

// create all possible pairs of Groups in a specific set
std::vector<std::pair<GroupIt, GroupIt>> combinations(Set *set);

// test if two doubles are equal
bool equal(double a, double b);

// test if a double is less or equal than an other double
bool lessOrEqual(double a, double b);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

namespace permutation {

typedef std::vector<Operation*>::iterator PermutationEdge;

// input: vector of Operations
// changes input: sort vector in BB regions and sort those with regular definition of "less" in Operation class
// output: starting and ending operations of every present basic block
/*
 * example: state: BB1{Op1, Op2}, BB2{Op3, Op4, Op5}
 *          input: {Op3, Op2, Op1, Op5, Op4}
 *          change to: {Op1, Op2, Op3, Op4, Op5}
 *          output: {0,2},{2,5} 
 */
void findBBEdges(std::deque<std::pair<int, int>>& BBops, std::vector<Operation*>& permutation_vector);

// inputs: permutation_vector.begin(), output of function findBBEdges
// changes: permutation_vector to the next permutation step
// output: false if all permuations visited, else true
/*
 * further information: This is a extended version of next_permutation in package algorithm.
 *                      As we do not need to permute operations of different basic blocks,
 *                      this function does exactly only permute within a BB region and
 *                      goes over all combinations of permutations of different BBs.
 */
bool get_next_permutation(PermutationEdge begin_of_permutation_vector, std::deque<std::pair<int, int>>& separation_of_BBs);

} // namespace permuation

namespace dynamatic {
namespace buffer {
namespace fpga20 {

class MyFPGA20Buffers : public FPGA20Buffers {
    public:
    std::vector<ResourceSharingInfo::OperationData> getData();
    double getOccupancySum(std::set<Operation*>& group);
    LogicalResult addSyncConstraints(std::vector<Value> opaqueChannel);

    //constructor
    MyFPGA20Buffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                            double targetPeriod, bool legacyPlacement, Logger &logger, StringRef milpName)
        : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement, logger, milpName){
    
        };
};

} // namespace fpga20
} // namespace buffer
} // namespace dynamatic

#endif //EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H