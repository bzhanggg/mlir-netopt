#include "DeadQueueElimination.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/dialect/SpmcOps.h"

#define DEBUG_TYPE "dead-queue-elimination"

namespace mlir {
namespace spmc {

#define GEN_PASS_DEF_DEADQUEUEELIMINATION
#include "src/transform/DeadQueueElimination/Passes.h.inc"

/// Check if a queue value is ever popped from.
static bool hasPopUses(Value queue) {
  return std::any_of(
      queue.getUses().begin(), queue.getUses().end(),
      [](OpOperand &use) { return isa<spmc::PopOp>(use.getOwner()); });
}

/// Check if a queue value escapes the function (e.g., returned or passed to
/// external calls).
static bool doesQueueEscape(Value queue) {
  return std::any_of(
      queue.getUses().begin(), queue.getUses().end(), [](OpOperand &use) {
        Operation *user = use.getOwner();
        return llvm::isa<func::ReturnOp>(user) || llvm::isa<func::CallOp>(user);
      });
}

/// Check if a queue is "write-only": created, has push ops, but no pop ops and
/// doesn't escape.
static bool isWriteOnlyQueue(Value queue) {
  unsigned pushCount = 0;
  unsigned popCount = 0;

  for (auto &use : queue.getUses()) {
    if (llvm::isa<spmc::PushOp>(use.getOwner())) {
      pushCount++;
    } else if (llvm::isa<spmc::PopOp>(use.getOwner())) {
      popCount++;
    }
  }
  return pushCount > 0 && popCount == 0 && !doesQueueEscape(queue);
}

struct RemoveDeadQueuePattern : public OpRewritePattern<spmc::CreateOp> {
  RemoveDeadQueuePattern(MLIRContext *context)
      : OpRewritePattern<spmc::CreateOp>(context) {};

  llvm::LogicalResult
  matchAndRewrite(spmc::CreateOp op, PatternRewriter &rewriter) const override {
    Value queueResult = op.getResult();

    // Case 1: zero uses
    if (!queueResult.hasNUsesOrMore(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Eliminating unused queue\n");
      rewriter.eraseOp(op);
      return llvm::success();
    }

    // Case 2: Write-only queue (pushed but never popped, doesn't escape)
    if (isWriteOnlyQueue(queueResult)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Eliminating write-only queue and its push operations\n");

      // Collect all push operations on this queue
      SmallVector<spmc::PushOp> pushOps;
      for (const OpOperand &use : queueResult.getUses()) {
        if (const spmc::PushOp pushOp =
                llvm::dyn_cast<spmc::PushOp>(use.getOwner())) {
          pushOps.emplace_back(pushOp);
        }
      }

      std::for_each(pushOps.begin(), pushOps.end(),
                    [&](const spmc::PushOp &op) { rewriter.eraseOp(op); });
      rewriter.eraseOp(op);
      return llvm::success();
    }
    return llvm::failure();
  }
};

struct DeadQueueElimination
    : impl::DeadQueueEliminationBase<DeadQueueElimination> {
  using DeadQueueEliminationBase::DeadQueueEliminationBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns{&getContext()};
    patterns.add<RemoveDeadQueuePattern>(&getContext());

    if (llvm::failed(
            applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace spmc
} // namespace mlir