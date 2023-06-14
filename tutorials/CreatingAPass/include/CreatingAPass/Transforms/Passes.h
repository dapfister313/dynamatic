//===- Passes.h - Transformation passes registration ------------*- C++ -*-===//
//
// This file contains declarations to register transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CREATINGAPASS_TRANSFORMS_PASSES_H
#define CREATINGAPASS_TRANSFORMS_PASSES_H

#include "CreatingAPass/Transforms/SimpleTransform.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {
namespace tutorials {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "CreatingAPass/Transforms/Passes.h.inc"

} // namespace tutorials
} // namespace dynamatic
#endif // CREATINGAPASS_TRANSFORMS_PASSES_H