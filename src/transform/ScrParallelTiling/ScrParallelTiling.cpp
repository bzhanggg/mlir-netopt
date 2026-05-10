#include "src/analysis/LoopStateAccessAnalysis/LoopStateAccessAnalysis.h"
#include "src/dialect/SpmcDialect.h"
#include "src/dialect/SpmcOps.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "scr-parallel-tile"

namespace mlir {
namespace spmc {

#define GEN_PASS_DECL_SCRPARALLELTILING
#define GEN_PASS_DEF_SCRPARALLELTILING
#include "src/transform/ScrParallelTiling/Passes.h.inc"

struct ScrParallelTiling : impl::ScrParallelTilingBase<ScrParallelTiling> {

  using ScrParallelTilingBase::ScrParallelTilingBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    LoopStateAccessAnalysis analysis{op};

    llvm::SmallVector<affine::AffineForOp> scrLoops;
    op->walk([&](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>() &&
          analysis.isScrCandidate(forOp)) {
        scrLoops.emplace_back(forOp);
      }
    });

    size_t transformed = 0;
    for (affine::AffineForOp forOp : scrLoops) {
      LLVM_DEBUG(llvm::dbgs() << "SCR tiling loop at " << forOp.getLoc()
                              << " with " << numCores << " cores\n");
      SmallVector<affine::AffineForOp, 1> nest{forOp};
      SmallVector<affine::AffineForOp> tiledNest;
      if (llvm::failed(
              affine::tilePerfectlyNested(nest, {numCores}, &tiledNest))) {
        LLVM_DEBUG(llvm::dbgs() << "  tilePerfectlyNested failed\n");
        continue;
      }
      affine::AffineForOp innerLoop = tiledNest.back();
      if (failed(affine::affineParallelize(innerLoop, {}))) {
        LLVM_DEBUG(llvm::dbgs() << "  affineParallelize failed\n");
        continue;
      }
      ++transformed;
    }

    LLVM_DEBUG(llvm::dbgs() << "SCR parallel-tiled " << transformed << " of "
                            << scrLoops.size() << " candidate loops\n");
  }
};

} // namespace spmc
} // namespace mlir
