#include "LoopStateAccessAnalysis.h"

#include "src/dialect/SpmcOps.h"

#include <mlir/Dialect/Affine/Analysis/AffineAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/Support/Debug.h>

#include <optional>

#define DEBUG_TYPE "loop-state-access-analysis"

namespace mlir {
namespace spmc {

LoopStateAccessAnalysis::LoopStateAccessAnalysis(Operation *op) {
  op->walk([&](affine::AffineForOp forOp) { analyzeLoop(forOp); });
}

void LoopStateAccessAnalysis::analyzeLoop(affine::AffineForOp forOp) {
  LoopAccessInfo info;

  checkSpmcOps(forOp, info);
  checkIterArgs(forOp, info);
  checkUnknownSideEffects(forOp, info);
  checkMemoryDependencies(forOp, info);

  info.isParallelizable = !info.hasSpmc && !info.hasIterArgs &&
                          !info.hasUnknownSideEffects &&
                          !info.hasMemoryConflicts;

  info.isScrCandidate = info.hasSpmcPop && !info.hasSpmcPush &&
                        !info.hasIterArgs && !info.hasUnknownSideEffects &&
                        !info.hasMemoryConflicts;

  LLVM_DEBUG({
    llvm::dbgs() << "Loop at " << forOp->getLoc() << ": ";
    if (info.isParallelizable) {
      llvm::dbgs() << "PARALLELIZABLE\n";
    } else {
      llvm::dbgs() << "NOT PARALLELIZABLE (";
      if (info.hasSpmc) {
        llvm::dbgs() << "spmc_ops ";
      }
      if (info.hasIterArgs) {
        llvm::dbgs() << "iter_args ";
      }
      if (info.hasUnknownSideEffects) {
        llvm::dbgs() << "unknown_effects ";
      }
      if (info.hasMemoryConflicts) {
        llvm::dbgs() << "memory_conflicts ";
      }
      llvm::dbgs() << ")\n";
    }
  });

  loopInfoMap.emplace_or_assign(forOp.getOperation(), std::move(info));
}

void LoopStateAccessAnalysis::checkSpmcOps(affine::AffineForOp forOp,
                                           LoopAccessInfo &info) {
  forOp.walk([&](Operation *op) {
    if (isa<spmc::PushOp>(op)) {
      info.hasSpmcPush = true;
      info.blockingOps.emplace_back(op);
    } else if (isa<spmc::PopOp>(op)) {
      info.hasSpmcPop = true;
      info.blockingOps.emplace_back(op);
    }
  });
  info.hasSpmc = info.hasSpmcPush || info.hasSpmcPop;
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
  forOp.walk([&](Operation *op) {
    if (op == forOp.getOperation()) {
      return WalkResult::advance();
    }
    if (isa<AffineForOp, AffineYieldOp, AffineIfOp, AffineLoadOp, AffineStoreOp,
            spmc::PushOp, spmc::PopOp>(op)) {
      return WalkResult::advance();
    }
    if (isMemoryEffectFree(op)) {
      return WalkResult::advance();
    }
    info.hasUnknownSideEffects = true;
    info.blockingOps.emplace_back(op);
    return WalkResult::advance();
  });
}

void LoopStateAccessAnalysis::checkMemoryDependencies(affine::AffineForOp forOp,
                                                      LoopAccessInfo &info) {
  if (info.hasIterArgs || info.hasUnknownSideEffects) {
    return;
  }
  if (!affine::isLoopMemoryParallel(forOp)) {
    info.hasMemoryConflicts = true;
  }
}

bool LoopStateAccessAnalysis::isParallelizable(
    affine::AffineForOp forOp) const {
  auto it = loopInfoMap.find(forOp.getOperation());
  if (it == loopInfoMap.end()) {
    return false;
  }
  return it->second.isParallelizable;
}

bool LoopStateAccessAnalysis::isScrCandidate(affine::AffineForOp forOp) const {
  auto it = loopInfoMap.find(forOp.getOperation());
  if (it == loopInfoMap.end()) {
    return false;
  }
  return it->second.isScrCandidate;
}

const std::optional<LoopAccessInfo>
LoopStateAccessAnalysis::getAccessInfo(affine::AffineForOp forOp) const {
  auto it = loopInfoMap.find(forOp.getOperation());
  if (it == loopInfoMap.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace spmc
} // namespace mlir