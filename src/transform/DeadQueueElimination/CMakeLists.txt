set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DeadQueueElimination)
add_public_tablegen_target(MLIRDeadQueueEliminationPasses)
add_mlir_doc(Passes DeadQueueElimination ./ -gen-pass-doc)

add_mlir_library(DeadQueueEliminationPasses
    DeadQueueElimination.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/transform/DeadQueueElimination
    ${PROJECT_SOURCE_DIR}/src/dialect

    DEPENDS
    MLIRDeadQueueEliminationPasses
    MLIRSpmc

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)
