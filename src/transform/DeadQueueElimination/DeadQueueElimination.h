#ifndef TRANSFORM_DEAD_QUEUE_ELIMINATION_H_
#define TRANSFORM_DEAD_QUEUE_ELIMINATION_H_

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace spmc {

#define GEN_PASS_DECL_DEADQUEUEELIMINATION
#include "src/transform/DeadQueueElimination/Passes.h.inc"

} // namespace spmc
} // namespace mlir

#endif // TRANSFORM_DEAD_QUEUE_ELIMINATION_H_