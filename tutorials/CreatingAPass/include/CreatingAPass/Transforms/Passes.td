//===- Passes.td - Transformation passes definition --------*- tablegen -*-===//
//
// This file contains the definition for the single transformation pass of this
// tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef CREATINGAPASS_TRANSFORMS_PASSES_TD
#define CREATINGAPASS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SimpleTransform : Pass< "dynamatic-tutorial-simple-transform", 
                            "mlir::ModuleOp"> {
  let summary = "This is summary information";
  let description = [{
    This is a longer and
    possibly multilined description of the pass.
  }];
  let constructor = "dynamatic::tutorials::createSimpleTransformPass()";
}

#endif // CREATINGAPASS_TRANSFORMS_PASSES_TD