#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H

#include <set>
#include <list>
#include <stack>
#include <vector>
#include <algorithm>

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "experimental/Support/StdProfiler.h"

namespace dynamatic {
namespace experimental {
namespace sharing {

// kosarajus algorithm performed on basic block level
std::vector<int> Kosarajus_algorithm_BBL(SmallVector<ArchBB> archs);

// different implementation: performed on operation level
void Kosarajus_algorithm_OPL(mlir::Operation* startOp, std::set<mlir::Operation*>& result);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SCC_H
