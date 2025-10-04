#include "SpmcDialect.h"

#include "SpmcOps.h"
#include "SpmcTypes.h"
#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

#include "src/dialect/SpmcDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "src/dialect/SpmcTypes.cpp.inc"
#define GET_OP_CLASSES
#include "src/dialect/SpmcOps.cpp.inc"

namespace mlir {
namespace spmc {

void SpmcDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "src/dialect/SpmcTypes.cpp.inc"
    >();
    addOperations<
#define GET_OP_LIST
#include "src/dialect/SpmcOps.cpp.inc"
    >();
}

} // namespace spmc
} // namespace mlir
