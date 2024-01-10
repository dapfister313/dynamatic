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

#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace sharing {

#define GEN_PASS_DECL_RESOURCESHARINGFCCM22
#define GEN_PASS_DEF_RESOURCESHARINGFCCM22
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createResourceSharingFCCM22Pass(StringRef algorithm = "fpga20",
                                StringRef frequencies = "",
                                StringRef timingModels = "",
                                bool firstCFDFC = false, double targetCP = 4.0,
                                unsigned timeout = 180, bool dumpLogs = false);

} // namespace sharing
} // namespace dynamatic

#endif // INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_FCCM22SHARING_H