#!/bin/bash

BUILD_SYSTEM="Ninja"
BUILD_DIR=./build

mkdir -p llvm-project/build
pushd llvm-project/build

cmake -G $BUILD_SYSTEM ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build . --parallel

popd


cmake --build $BUILD_DIR --target mlir-headers
cmake --build $BUILD_DIR --target mlir-doc
