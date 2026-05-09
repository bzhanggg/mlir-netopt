#include "src/analysis/LoopStateAccessAnalysis/LoopStateAccessAnalysis.h"
#include "src/dialect/SpmcDialect.h"
#include "src/dialect/SpmcOps.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "state-ind-parallel"

namespace mlir {
namespace spmc {

#define GEN_PASS_DEF_STATEINDEPENDENTPARALLELIZATION
#include "src/transform/StateIndependentParallelization/Passes.h.inc"

struct StateIndependentParallelization
    : impl::StateIndependentParallelizationBase<
          StateIndependentParallelization> {
  using StateIndependentParallelizationBase::
      StateIndependentParallelizationBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    LoopStateAccessAnalysis analysis{op};

    llvm::SmallVector<affine::AffineForOp> outerLoops;
    op->walk([&](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>()) {
        outerLoops.push_back(forOp);
      }
    });

    unsigned parallelized = 0;
    for (affine::AffineForOp forOp : outerLoops) {
      if (!analysis.isParallelizable(forOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping non-parallelizable loop at "
                                << forOp.getLoc() << "\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Parallelizing loop at " << forOp.getLoc() << "\n");
      if (failed(affine::affineParallelize(forOp, {}))) {
        LLVM_DEBUG(llvm::dbgs() << "  affineParallelize failed\n");
        continue;
      }
      ++parallelized;
    }
    LLVM_DEBUG(llvm::dbgs() << "Parallelized " << parallelized << " of "
                            << outerLoops.size() << " outermost loops\n");
  }
};

} // namespace spmc
} // namespace mlir