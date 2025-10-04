set(LLVM_TARGET_DEFINITIONS SpmcOps.td)

# operations
mlir_tablegen(SpmcOps.h.inc -gen-op-decls)
mlir_tablegen(SpmcOps.cpp.inc -gen-op-defs)
# types
mlir_tablegen(SpmcTypes.h.inc -gen-typedef-decls -typedefs-dialect=spmc)
mlir_tablegen(SpmcTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=spmc)
# dialect
mlir_tablegen(SpmcDialect.h.inc -gen-dialect-decls -dialect=spmc)
mlir_tablegen(SpmcDialect.cpp.inc -gen-dialect-defs -dialect=spmc)
# tablegen
add_public_tablegen_target(SpmcOpsIncGen)
add_dependencies(mlir-headers SpmcOpsIncGen)

add_mlir_dialect_library(MLIRSpmc
    SpmcDialect.cpp
    SpmcOps.cpp
    DEPENDS
        SpmcOpsIncGen

    LINK_LIBS PUBLIC
)
