#include "LoopStateAccessAnalysis.h"

#include "src/dialect/SpmcOps.h"

#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

namespace mlir {
namespace spmc {

LoopStateAccessAnalysis::LoopStateAccessAnalysis(Operation *op) {
  op->walk([&](affine::AffineForOp forOp) { analyzeLoop(forOp); });
}

void LoopStateAccessAnalysis::analyzeLoop(affine::AffineForOp forOp) {
  LoopAccessInfo info;
}

void LoopStateAccessAnalysis::checkSpmcOps(affine::AffineForOp forOp,
                                           LoopAccessInfo &info) {
  forOp.walk([&](Operation *op) {
    if (isa<spmc::PushOp, spmc::PopOp>(op)) {
      info.hasSpmc = true;
      info.blockingOps.push_back(op);
    }
  });
}

void LoopStateAccessAnalysis::checkIterArgs(affine::AffineForOp forOp,
                                            LoopAccessInfo &info) {
  if (forOp.getNumIterOperands() > 0) {
    info.hasIterArgs = true;
  }
}

void LoopStateAccessAnalysis::checkUnknownSideEffects(affine::AffineForOp forOp,
                                            LoopAccessInfo &info) {
                                                using namespace affine;
    forOp.walk([&](Operation* op) {
        if (op == forOp.getOperation()) {
            return WalkResult::advance();
        }
        if (isa<AffineForOp, AffineYieldOp, AffineIfOp, AffineLoadOp, AffineStoreOp>(op)) {
            return WalkResult::advance();
        }
        if (isMemoryEffectFree(op)) {
            return WalkResult::advance();
        }
        info.hasUnknownSideEffects = true;
        info.blockingOps.push_back(op);
        return WalkResult::advance();
    });
}

void LoopStateAccessAnalysis::checkMemoryDependencies(affine::AffineForOp forOp, LoopAccessInfo& info) {
    if (info.hasSpmc || info.hasIterArgs || info.hasUnknownSideEffects) {
        return;
    }
    if (!affine::isLoopMemoryParallel(forOp)) {
        info.hasMemoryConflicts = true;
    }
}

bool LoopStateAccessAnalysis::isParallelizable(affine::AffineForOp forOp) const {
    auto it = loopInfoMap.find(forOp.getOperation());
    if (it == loopInfoMap.end()) {
        return false;
    }
    return it->second.isParallelizable;
}

const std::optional<LoopAccessInfo> LoopStateAccessAnalysis::getAccessInfo(affine::AffineForOp forOp) const {
    auto it = loopInfoMap.find(forOp.getOperation());
    if (it == loopInfoMap.end()) {
        return std::nullopt;
    }
    return it->second;
}

} // namespace spmc
} // namespace mlir