add_mlir_library(SpmcTransforms
    DeadQueueElimination.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/transform

    LINK_LIBS PUBLIC
    MLIRSpmc
    MLIRPass
    MLIRIR
)

set(LLVM_TARGET_DEFINITIONS Transforms.td)
mlir_tablegen(Transforms.h.inc -gen-pass-decls -name SpmcTransforms)
# add_mlir_doc(SpmcTransforms ./ -gen-pass-doc)
