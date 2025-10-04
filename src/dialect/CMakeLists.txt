set(LLVM_LINK_COMPONENTS Support)

# Operations tablegen
set(LLVM_TARGET_DEFINITIONS SpmcOps.td)
mlir_tablegen(SpmcOps.h.inc -gen-op-decls)
mlir_tablegen(SpmcOps.cpp.inc -gen-op-defs)
mlir_tablegen(SpmcDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SpmcDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(SpmcOpsIncGen)

add_mlir_dialect_library(MLIRSpmc
    SpmcDialect.cpp

    DEPENDS SpmcOpsIncGen
    LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
)
