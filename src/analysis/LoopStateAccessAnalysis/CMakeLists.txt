add_mlir_library(LoopStateAccessAnalysis
    LoopStateAccessAnalysis.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/analysis/LoopStateAccessAnalysis

    LINK_LIBS PUBLIC
)