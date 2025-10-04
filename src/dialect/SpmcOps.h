#ifndef SPMC_OPS_H_
#define SPMC_OPS_H_

#include "SpmcDialect.h"
#include "SpmcTypes.h"
#include <mlir/Bytecode/BytecodeOpInterface.h>

#define GET_OP_CLASSES
#include "src/dialect/SpmcOps.h.inc"

#endif