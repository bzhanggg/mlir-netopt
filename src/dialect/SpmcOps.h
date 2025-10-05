#ifndef SPMC_OPS_H_
#define SPMC_OPS_H_

#include "SpmcDialect.h"
#include "SpmcTypes.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

#define GET_OP_CLASSES
#include "src/dialect/SpmcOps.h.inc"

#endif // SPMC_OPS_H_