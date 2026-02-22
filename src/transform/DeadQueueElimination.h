#ifndef TRANSFORM_DEAD_QUEUE_ELIMINATION_H_
#define TRANSFORM_DEAD_QUEUE_ELIMINATION_H_

#include <mlir/Pass/Pass.h>

namespace mlir {
namespace spmc {

class DeadQueueElimination
    : public PassWrapper<DeadQueueElimination, OperationPass<>> {
public:
  void runOnOperation() override;

private:
  StringRef getArgument() const final { return "dead-queue-elimination"; }
  StringRef getDescription() const final { return "Eliminate dead queues"; }

  /// Walks all operations in a region and marks queues used by push/pop as
  /// live
  void markLiveQueues(Operation *op, llvm::SmallPtrSet<Value, 16> &liveQueues);

  /// Checks if a value is a queue operand in any push_back or pop_front op
  bool isQueueUsed(Value queue, Operation *root);

  /// Collects all spmc.create ops in the op tree
  void collectSpmcCreateOps(Operation *op,
                            const llvm::SmallPtrSet<Value, 16> &liveQueues,
                            llvm::SmallVector<Operation *, 16> &deadOps);

  /// Erases dead ops
  void eliminateDeadQueues(llvm::SmallVector<Operation *, 16> &deadOps);

  /// Safety check to eliminate queues iff not used by non-spmc ops
  bool hasNonSpmcUses(Value queue);

  /// Diagnostic print removed queue info
  void reportRemovedQueue(Operation *createOp);

  /// Stats for debugging
  unsigned numQueuesEliminated = 0;
};

} // namespace spmc
} // namespace mlir

#endif // TRANSFORM_DEAD_QUEUE_ELIMINATION_H_