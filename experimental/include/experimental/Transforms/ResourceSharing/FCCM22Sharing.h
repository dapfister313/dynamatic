//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --resource-sharing-FCCM22 pass.
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H
#define INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "dynamatic/Support/CFG.h"


namespace dynamatic {
namespace experimental {
namespace sharing {

//stores/transfers information needed for resource sharing
struct ResourceSharingInfo {

// for each CFDFC, store the throughput in double format to 
// double format to compare 
std::map<int, double> sharing_check{};

//store stats of each operation
struct OpSpecific {
    mlir::Operation* op; 
    double occupancy; 
    double op_latency;
    //double throughput;

    void print() {
    llvm::errs() << "Operation " << op 
                << ", occupancy: " << occupancy 
                << ", latency: " << op_latency 
                << ", block: " << getLogicBB(op)
                << "\n";
    }
};
std::vector<OpSpecific> sharing_init;

//constructor
ResourceSharingInfo() = default;
};

#define GEN_PASS_DECL_RESOURCESHARINGFCCM22
#define GEN_PASS_DEF_RESOURCESHARINGFCCM22
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createResourceSharingFCCM22Pass(StringRef algorithm = "fpga20",
                                StringRef frequencies = "",
                                StringRef timingModels = "",
                                bool firstCFDFC = false, double targetCP = 4.0,
                                unsigned timeout = 180, bool dumpLogs = false);

} // namespace sharing
} // namespace experimental
} // namespace dynamatic

#endif // INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H