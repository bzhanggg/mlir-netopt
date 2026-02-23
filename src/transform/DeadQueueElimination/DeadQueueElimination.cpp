#include "DeadQueueElimination.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/dialect/SpmcDialect.h"
#include "src/dialect/SpmcOps.h"

#define DEBUG_TYPE "dead-queue-elimination"

namespace mlir {
namespace spmc {

#define GEN_PASS_DEF_DEADQUEUEELIMINATION
#include "src/transform/DeadQueueElimination/Passes.h.inc"

struct RemoveDeadQueuePattern : public OpRewritePattern<spmc::CreateOp> {
    RemoveDeadQueuePattern(MLIRContext *context) : OpRewritePattern<spmc::CreateOp>(context) {};

    LogicalResult matchAndRewrite(spmc::CreateOp op, PatternRewriter &rewriter) const override {
        if (!op.getResult().hasNUsesOrMore(1)) {
            rewriter.eraseOp(op);
            return llvm::success();
        }
        return llvm::failure();
    }
};

struct DeadQueueElimination : impl::DeadQueueEliminationBase<DeadQueueElimination> {
    using DeadQueueEliminationBase::DeadQueueEliminationBase;

    void runOnOperation() {
        mlir::RewritePatternSet patterns{&getContext()};
        patterns.add<RemoveDeadQueuePattern>(&getContext());

        if (llvm::failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace spmc
} // namespace mlir