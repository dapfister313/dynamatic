//===- SimpleTranform.h - Simple IR transformation pass ---------*- C++ -*-===//
//
// This file declares the --dynamatic-tutorial-simple-transform pass.
//
//===----------------------------------------------------------------------===//

#ifndef CREATEAPASS_TRANSFORMS_SIMPLETRANSFORM_H
#define CREATEAPASS_TRANSFORMS_SIMPLETRANSFORM_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace dynamatic {
namespace tutorials {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createSimpleTransformPass();

} // namespace tutorials
} // namespace dynamatic

#endif // CREATEAPASS_TRANSFORMS_SIMPLETRANSFORM_H