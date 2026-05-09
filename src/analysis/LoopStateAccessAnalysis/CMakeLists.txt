add_mlir_library(LoopStateAccessAnalysis
    LoopStateAccessAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/analysis/LoopStateAccessAnalysis
    ${PROJECT_SOURCE_DIR}/src/dialect

    DEPENDS
    MLIRSpmc

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRSideEffectInterfaces
)