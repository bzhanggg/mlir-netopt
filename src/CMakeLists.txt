add_subdirectory(dialect)
add_subdirectory(transform)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create compiler executable
add_llvm_executable(spmcc spmcc.cpp)

llvm_update_compile_flags(spmcc)
target_link_libraries(
    spmcc
    PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        DeadQueueEliminationPasses
        MLIRSpmc
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRPass
        MLIROptLib
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
)

mlir_check_all_link_libraries(spmcc)