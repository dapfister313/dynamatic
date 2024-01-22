add_dynamatic_library(DynamaticTutorialsMyCreatingPasses
  FCCM22Sharing.cpp
  SCC.cpp
  modIR.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)