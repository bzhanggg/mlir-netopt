#include "DeadQueueElimination.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

#define DEBUG_TYPE "dead-queue-elimination"

namespace mlir {
namespace spmc {

#define GEN_PASS_DEF_DEADQUEUEELIMINATION
#include "src/transform/DeadQueueElimination/Passes.h.inc"

} // namespace spmc
} // namespace mlir