add_dynamatic_library(DynamaticTutorialCreatingAPass
  SimpleTransform.cpp

  DEPENDS
  DynamaticTutorialCreatingAPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)
