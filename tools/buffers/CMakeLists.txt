set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(buffers
buffers.cpp
)

llvm_update_compile_flags(buffers)
target_link_libraries(buffers
  PRIVATE
  DynamaticSupport
  DynamaticTransforms
  DynamaticBufferPlacement

  CIRCTHandshake
  CIRCTHandshakeTransforms

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRMemRefDialect
  )
