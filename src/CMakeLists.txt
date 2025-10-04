add_subdirectory(dialect)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create compiler executable
add_executable(spmcc spmcc.cpp)

llvm_update_compile_flags(spmcc)
target_link_libraries(
    spmcc
    PRIVATE MLIRSpmc
            MLIRAnalysis
            MLIRIR
            MLIRParser
            MLIRPass
)

mlir_check_link_libraries(spmcc)