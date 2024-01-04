//===- FCCM22Sharing.h - resource-sharing -----*- C++ -*-===//
//
// Implements the --resource-sharing pass, which checks for sharable
// Operations (little to none performance overhead).
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ResourceSharing/FCCM22Sharing.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace circt;

namespace {

struct ResourceSharingFCCM22Pass
    : public dynamatic::sharing::impl::ResourceSharingFCCM22Base<
          ResourceSharingFCCM22Pass> {

  void runOnOperation() override;
};
} // namespace

void ResourceSharingFCCM22Pass::runOnOperation() {
  llvm::errs() << "Borderplate code to determine if pass runs as expected\n";
}

namespace dynamatic {
namespace sharing {

/// Returns a unique pointer to an operation pass that matches MLIR modules.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createResourceSharingFCCM22Pass() {
  return std::make_unique<ResourceSharingFCCM22Pass>();
}
} // namespace sharing
} // namespace dynamatic