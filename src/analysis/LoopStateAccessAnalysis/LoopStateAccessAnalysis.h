#ifndef SRC_ANALYSIS_LOOP_STATE_ACCESS_ANALYSIS_H_
#define SRC_ANALYSIS_LOOP_STATE_ACCESS_ANALYSIS_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/Operation.h>

#include <optional>

namespace mlir {
namespace spmc {

struct LoopAccessInfo {
  bool hasSpmc = false;
  bool hasSpmcPush = false;
  bool hasSpmcPop = false;
  bool hasIterArgs = false;
  bool hasUnknownSideEffects = false;
  bool hasMemoryConflicts = false;
  bool isParallelizable = false;
  bool isScrCandidate = false;

  llvm::SmallVector<const Operation *, 4> blockingOps;
};

class LoopStateAccessAnalysis {
public:
  explicit LoopStateAccessAnalysis(Operation *op);

  bool isParallelizable(affine::AffineForOp forOp) const;
  bool isScrCandidate(affine::AffineForOp forOp) const;
  const std::optional<LoopAccessInfo>
  getAccessInfo(affine::AffineForOp forOp) const;

private:
  void analyzeLoop(affine::AffineForOp);
  void checkSpmcOps(affine::AffineForOp forOp, LoopAccessInfo &info);
  void checkIterArgs(affine::AffineForOp forOp, LoopAccessInfo &info);
  void checkUnknownSideEffects(affine::AffineForOp forOp, LoopAccessInfo &info);
  void checkMemoryDependencies(affine::AffineForOp forOp, LoopAccessInfo &info);

  llvm::DenseMap<Operation *, LoopAccessInfo> loopInfoMap;
};

} // namespace spmc
} // namespace mlir

#endif // SRC_ANALYSIS_LOOP_STATE_ACCESS_ANALYSIS_H_