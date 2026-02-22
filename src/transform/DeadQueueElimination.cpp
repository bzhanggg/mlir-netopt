#include "DeadQueueElimination.h"

#include <llvm/Support/Debug.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#define DEBUG_TYPE "dead-queue-elimination"

namespace mlir {
namespace spmc {

void DeadQueueElimination::runOnOperation() {
  Operation *op = getOperation();
}

} // namespace spmc
} // namespace mlir