add_dynamatic_library(DynamaticTransforms
  AnalyzeMemoryAccesses.cpp
  FlattenMemRefRowMajor.cpp
  HandshakePrepareForLegacy.cpp
  HandshakeInferBasicBlocks.cpp
  HandshakeSimplify.cpp
  NameMemoryOps.cpp
  PushConstants.cpp

  DEPENDS
  DynamaticTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)
