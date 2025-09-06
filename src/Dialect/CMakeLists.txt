set(LLVM_TARGET_DEFINITIONS PolyOps.td)

add_mlir_dialect_library(MLIRPoly

    LINK_LIBS PUBLIC
)