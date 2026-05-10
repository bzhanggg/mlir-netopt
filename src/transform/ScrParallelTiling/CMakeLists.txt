set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ScrParallelTiling)
add_public_tablegen_target(MLIRScrParallelTilingPasses)
add_mlir_doc(Passes ScrParallelTiling ./ -gen-pass-doc)

add_mlir_library(ScrParallelTilingPasses
    ScrParallelTiling.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/transform/ScrParallelTiling
    ${PROJECT_SOURCE_DIR}/src/dialect
    ${PROJECT_SOURCE_DIR}/src/analysis/LoopStateAccessAnalysis

    DEPENDS
    MLIRScrParallelTilingPasses
    MLIRSpmc

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRAffineDialect
    MLIRAffineUtils
    LoopStateAccessAnalysis
)
