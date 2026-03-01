set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StateIndependentParallelization)
add_public_tablegen_target(MLIRStateIndependentParallelizationPasses)
add_mlir_doc(Passes StateIndependentParallelization ./ -gen-pass-doc)

add_mlir_library(StateIndependentParallelizationPasses
    StateIndependentParallelization.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/transform/StateIndependentParallelization
    ${PROJECT_SOURCE_DIR}/src/dialect

    DEPENDS
    MLIRStateIndependentParallelizationPasses
    MLIRSpmc

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)
