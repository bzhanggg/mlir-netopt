cmake_minimum_required(VERSION 3.20.0)

project(spmc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

# Set LLVM and MLIR paths - adjust these paths to your llvm-project build directory
set(LLVM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/build/lib/cmake/mlir)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Set paths to LLVM and MLIR
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# Include essential LLVM and MLIR CMake functions
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Add LLVM and MLIR headers to include path
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

# Add definitions required by LLVM/MLIR headers
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

# Set LLVM components needed
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
set(LLVM_ENABLE_ASSERTIONS ON)

# Configure output directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Link MLIR libraries
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/llvm-project)
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(src)
