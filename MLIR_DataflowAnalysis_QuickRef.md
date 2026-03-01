# MLIR Dataflow Analysis - Quick Reference & Code Snippets

## Quick Navigation Guide

### Choosing Your Analysis Type

```
Need to track:
├── SSA value properties? → Use SPARSE
│   ├── Going forward (from operands to results)? → SparseForwardDataFlowAnalysis
│   └── Going backward (from uses to definitions)? → SparseBackwardDataFlowAnalysis
│
└── Program state at points? → Use DENSE
    ├── Going forward (tracking state changes)? → DenseForwardDataFlowAnalysis
    └── Going backward (tracking reaching definitions)? → DenseBackwardDataFlowAnalysis
```

---

## 1. Minimal Example: Constant Value Tracker

```cpp
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace {
  using namespace mlir;
  using namespace mlir::dataflow;

  // Lattice value: does this SSA value have a known constant?
  struct ConstantInfo {
    std::optional<int64_t> value;  // nullopt = unknown

    static ConstantInfo join(const ConstantInfo &lhs,
                             const ConstantInfo &rhs) {
      // If both have same constant, return it; else unknown
      if (lhs.value == rhs.value)
        return lhs;
      return ConstantInfo{std::nullopt};
    }

    bool operator==(const ConstantInfo &other) const {
      return value == other.value;
    }

    void print(raw_ostream &os) const {
      if (value)
        os << *value;
      else
        os << "unknown";
    }
  };

  class ConstantTracker
      : public SparseForwardDataFlowAnalysis<Lattice<ConstantInfo>> {
  public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    LogicalResult visitOperation(Operation *op,
                                 ArrayRef<const Lattice<ConstantInfo> *> operands,
                                 ArrayRef<Lattice<ConstantInfo> *> results) override {
      // Handle arith.constant
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          ConstantInfo info{intAttr.getValue().getZExtValue()};
          results[0]->join(info);
        }
      }
      // Handle arith.addi: if both operands are constant, result is too
      else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        if (operands.size() == 2 && operands[0] && operands[1]) {
          if (operands[0]->getValue().value && operands[1]->getValue().value) {
            int64_t sum = *operands[0]->getValue().value +
                         *operands[1]->getValue().value;
            results[0]->join(ConstantInfo{sum});
          }
        }
      }
      return success();
    }

    void setToEntryState(Lattice<ConstantInfo> *lattice) override {
      propagateIfChanged(lattice, lattice->join(ConstantInfo{std::nullopt}));
    }
  };
}

// Use it
void analyzeFunction(func::FuncOp func) {
  DataFlowSolver solver;
  solver.load<ConstantTracker>();
  if (failed(solver.initializeAndRun(func)))
    return;

  func.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      auto *lattice = solver.lookupState<Lattice<ConstantInfo>>(result);
      if (lattice && lattice->getValue().value) {
        llvm::outs() << "Result is constant: "
                     << *lattice->getValue().value << "\n";
      }
    }
  });
}
```

---

## 2. Minimal Example: Liveness Analysis (Pre-built)

```cpp
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

using namespace mlir::dataflow;

void analyzeLiveness(Operation *module) {
  DataFlowSolver solver;
  solver.load<LivenessAnalysis>();

  if (failed(solver.initializeAndRun(module))) {
    llvm::errs() << "Analysis failed\n";
    return;
  }

  // Query results
  module->walk([&](Value value) {
    const auto *liveness = solver.lookupState<Liveness>(value);
    if (liveness && liveness->isLive) {
      llvm::outs() << "Value is live: " << value << "\n";
    }
  });
}
```

---

## 3. Minimal Example: Dead Code Analysis + Integer Range

```cpp
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

using namespace mlir::dataflow;

void analyzeRanges(Operation *module) {
  DataFlowSolver solver;
  // Load DeadCodeAnalysis first (dependency)
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();

  if (failed(solver.initializeAndRun(module))) {
    llvm::errs() << "Analysis failed\n";
    return;
  }

  module->walk([&](Value value) {
    // Check if block is live
    Block *block = value.getParentBlock();
    auto *executable = solver.lookupState<Executable>(block);
    if (!executable || !executable->isLive())
      return;

    // Get integer range
    const auto *rangeLattice =
        solver.lookupState<IntegerValueRangeLattice>(value);

    if (rangeLattice && !rangeLattice->getValue().isUninitialized()) {
      const ConstantIntRanges &ranges = rangeLattice->getValue().getValue();
      llvm::outs() << "Value range: [" << ranges.getSMin() << ", "
                   << ranges.getSMax() << "]\n";

      // Check if it's constant
      if (auto constVal = ranges.getConstantValue()) {
        llvm::outs() << "  -> Constant: " << constVal << "\n";
      }
    }
  });
}
```

---

## 4. Integrating Affine Analysis

```cpp
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"

using namespace mlir::affine;

void analyzeAffineLoop(AffineForOp forOp) {
  // Build constraint system from loop
  FlatAffineValueConstraints constraints;

  LogicalResult result = constraints.addAffineForOpDomain(forOp);
  if (failed(result)) {
    llvm::errs() << "Could not add loop constraints\n";
    return;
  }

  llvm::outs() << "Loop constraints:\n";
  constraints.print(llvm::outs());

  // Check if nested
  SmallVector<AffineForOp, 4> enclosingLoops;
  getAffineForIVs(*forOp.getBody()->front(), &enclosingLoops);

  llvm::outs() << "Nesting depth: " << getNestingDepth(forOp.getOperation())
               << "\n";
}
```

---

## 5. Composition Pattern: Integer Range + Dead Code

```cpp
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"

using namespace mlir::dataflow;

void optimizeArithmetic(func::FuncOp func) {
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();

  if (failed(solver.initializeAndRun(func))) {
    return;
  }

  func.walk([&](arith::DivSIOp divOp) {
    // Get divisor range
    auto *divisorRange =
        solver.lookupState<IntegerValueRangeLattice>(divOp.getRhs());

    if (!divisorRange)
      return;

    const ConstantIntRanges &range =
        divisorRange->getValue().getValue();

    // Check if divisor can be zero
    if (range.getSMin() <= 0 && range.getSMax() >= 0) {
      llvm::outs() << "Division by zero possible\n";
    } else {
      llvm::outs() << "Division is safe, could optimize\n";
    }
  });
}
```

---

## 6. Building Custom Sparse Forward Analysis (Complete)

```cpp
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace {
  using namespace mlir;
  using namespace mlir::dataflow;

  // Simple sign lattice: negative, zero, positive, unknown
  struct SignInfo {
    enum Sign { Negative, Zero, Positive, Unknown };
    Sign sign = Unknown;

    static SignInfo join(const SignInfo &lhs, const SignInfo &rhs) {
      // If either is unknown, result is unknown
      if (lhs.sign == Unknown || rhs.sign == Unknown)
        return SignInfo{Unknown};
      // If different signs, unknown
      if (lhs.sign != rhs.sign)
        return SignInfo{Unknown};
      return lhs;
    }

    bool operator==(const SignInfo &other) const {
      return sign == other.sign;
    }

    void print(raw_ostream &os) const {
      switch (sign) {
      case Negative: os << "negative"; break;
      case Zero: os << "zero"; break;
      case Positive: os << "positive"; break;
      case Unknown: os << "unknown"; break;
      }
    }
  };

  class SignAnalysis : public SparseForwardDataFlowAnalysis<Lattice<SignInfo>> {
  public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    LogicalResult visitOperation(Operation *op,
                                 ArrayRef<const Lattice<SignInfo> *> operands,
                                 ArrayRef<Lattice<SignInfo> *> results)
        override {
      // Constant
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          int64_t val = intAttr.getValue().getSExtValue();
          SignInfo::Sign sign = val < 0 ? SignInfo::Negative :
                               val > 0 ? SignInfo::Positive :
                               SignInfo::Zero;
          results[0]->join(SignInfo{sign});
        }
      }
      // Negation
      else if (auto negOp = dyn_cast<arith::NegFOp>(op)) {
        if (operands[0]) {
          SignInfo negated = operands[0]->getValue();
          if (negated.sign == SignInfo::Positive)
            negated.sign = SignInfo::Negative;
          else if (negated.sign == SignInfo::Negative)
            negated.sign = SignInfo::Positive;
          results[0]->join(negated);
        }
      }
      // Multiply
      else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        if (operands[0] && operands[1]) {
          SignInfo lhs = operands[0]->getValue();
          SignInfo rhs = operands[1]->getValue();
          if (lhs.sign != SignInfo::Unknown && rhs.sign != SignInfo::Unknown) {
            SignInfo::Sign resultSign =
                (lhs.sign == rhs.sign) ? SignInfo::Positive : SignInfo::Negative;
            if (lhs.sign == SignInfo::Zero || rhs.sign == SignInfo::Zero)
              resultSign = SignInfo::Zero;
            results[0]->join(SignInfo{resultSign});
          }
        }
      }
      return success();
    }

    void setToEntryState(Lattice<SignInfo> *lattice) override {
      propagateIfChanged(lattice, lattice->join(SignInfo{SignInfo::Unknown}));
    }
  };
}
```

---

## 7. Accessing Analysis Results Safely

```cpp
template <typename StateT>
const StateT *getOrNull(DataFlowSolver &solver, Value value) {
  const StateT *state = solver.lookupState<StateT>(value);
  return state;
}

template <typename StateT, typename ValueT>
std::optional<ValueT> getStateValue(DataFlowSolver &solver, Value value) {
  const auto *state = solver.lookupState<Lattice<ValueT>>(value);
  if (!state || state->getValue().isUninitialized())
    return std::nullopt;
  return state->getValue().getValue();
}

// Usage
DataFlowSolver solver;
solver.load<SomeAnalysis>();
if (failed(solver.initializeAndRun(op)))
  return;

// Safe query
if (auto val = getStateValue<MyLatticeTy, MyValueType>(solver, value)) {
  // Use *val
}
```

---

## 8. State Invalidation Pattern

```cpp
class AnalysisSyncListener : public RewriterBase::Listener {
private:
  DataFlowSolver &solver;

public:
  AnalysisSyncListener(DataFlowSolver &s) : solver(s) {}

  void notifyOperationErased(Operation *op) override {
    // Invalidate program point states
    if (auto pp = solver.getProgramPointAfter(op))
      solver.eraseState(pp);
    if (auto pp = solver.getProgramPointBefore(op))
      solver.eraseState(pp);

    // Invalidate SSA value states
    for (Value res : op->getResults())
      solver.eraseState(res);
  }

  void notifyOperationModified(Operation *op) override {
    // Invalidate states affected by modification
    solver.eraseState(solver.getProgramPointAfter(op));
    for (Value res : op->getResults())
      solver.eraseState(res);
  }

  void notifyBlockErased(Block *block) override {
    if (auto pp = solver.getProgramPointAfter(block))
      solver.eraseState(pp);
    if (auto pp = solver.getProgramPointBefore(block))
      solver.eraseState(pp);
  }
};

// Use in transformation
void transformWithAnalysis(Operation *op) {
  DataFlowSolver solver;
  solver.load<MyAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return;

  AnalysisSyncListener listener(solver);
  RewritePatternSet patterns(op->getContext());
  patterns.add<MyPattern>(...);

  (void)applyPatternsAndFoldGreedily(op, std::move(patterns),
                                      GreedyRewriteConfig().setListener(&listener));
}
```

---

## 9. Querying Dense Analysis Results

```cpp
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"

// Dense analysis attaches lattice to operations/blocks
class MyDenseAnalysis : public DenseForwardDataFlowAnalysis<DenseLattice<MyState>> {
  // implementation...
};

// Query at entry of block
const DenseLattice<MyState> *blockEntry =
  solver.lookupState<DenseLattice<MyState>>(block);

// Query at a program point (before operation)
ProgramPoint *pp = solver.getProgramPointBefore(op);
const DenseLattice<MyState> *state =
  solver.lookupState<DenseLattice<MyState>>(pp);

// Reconstruct state at specific point
module->walk([&](Operation *op) {
  ProgramPoint *before = solver.getProgramPointBefore(op);
  ProgramPoint *after = solver.getProgramPointAfter(op);

  if (const auto *beforeState =
      solver.lookupState<DenseLattice<MyState>>(before)) {
    llvm::outs() << "State before op: ";
    beforeState->print(llvm::outs());
  }
});
```

---

## 10. Affine Loop Constraint Queries

```cpp
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"

void analyzeLoopIterationSpace(AffineForOp forOp) {
  affine::FlatAffineValueConstraints cst;

  if (failed(cst.addAffineForOpDomain(forOp))) {
    llvm::errs() << "Could not analyze loop\n";
    return;
  }

  // Get dimension count
  unsigned numDims = cst.getNumDimVars();
  unsigned numSymbols = cst.getNumSymbolVars();

  llvm::outs() << "Iteration space has " << numDims << " dimensions "
               << "and " << numSymbols << " symbols\n";

  // Print all constraints
  cst.print(llvm::outs());

  // Check if a specific point is in iteration space
  // (Would need Presburger API for point-in-polytope checks)
}

void analyzeNestedLoops(AffineForOp outerLoop) {
  affine::FlatAffineValueConstraints cst;

  // Add outer loop
  cst.addAffineForOpDomain(outerLoop);

  // Add nested loop
  for (auto innerLoop : outerLoop.getOps<AffineForOp>()) {
    if (failed(cst.addAffineForOpDomain(innerLoop))) {
      llvm::errs() << "Could not add nested loop constraints\n";
      return;
    }

    // Now cst has constraints for both loops
    // Can query feasible iteration pairs
    llvm::outs() << "Feasible iteration space:\n";
    cst.print(llvm::outs());
  }
}
```

---

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| `lookupState` returns null | State not created/initialized | Check if analysis was loaded; run `initializeAndRun` |
| States not updating | Missed dependency | Ensure dependent state is accessed with dependency |
| Fixpoint not reached | Lattice doesn't monotonically increase | Check `join` is monotonic: `join(join(x,y),z) == join(x,join(y,z))` |
| Memory leak | States not erased after IR changes | Use `DataFlowListener` to keep solver in sync |
| Analysis too slow | Computing too many states | Make analysis more selective; use dead code pruning |
| Uninitialized states remain | Some transitions not modeled | Check all operation types are handled; use `visit` to initialize |

---

## File Checklist for Integration

- [ ] Include `mlir/Analysis/DataFlowFramework.h` - Core framework
- [ ] Include `mlir/Analysis/DataFlow/SparseAnalysis.h` - Sparse analyses
- [ ] Include `mlir/Analysis/DataFlow/DenseAnalysis.h` - Dense analyses
- [ ] Include `mlir/Analysis/DataFlow/DeadCodeAnalysis.h` - Used as dependency
- [ ] Include `mlir/Analysis/DataFlow/IntegerRangeAnalysis.h` - Range inference (if needed)
- [ ] Include `mlir/Dialect/Affine/Analysis/AffineStructures.h` - Affine constraints (if needed)
- [ ] Include `mlir/Dialect/Affine/Analysis/Utils.h` - Loop utilities (if needed)

---

## API Summary Table

| Class | Purpose | Use |
|-------|---------|-----|
| `DataFlowSolver` | Orchestrator | Load analyses, run, query results |
| `SparseForwardDataFlowAnalysis<StateT>` | Template for sparse forward | Extend for SSA value analysis (operands→results) |
| `SparseBackwardDataFlowAnalysis<StateT>` | Template for sparse backward | Extend for reaching uses analysis |
| `DenseForwardDataFlowAnalysis<StateT>` | Template for dense forward | Extend for state at program points |
| `Lattice<ValueT>` | Container for lattice value | Holds state for sparse SSA values |
| `DenseLattice<ValueT>` | Container for dense lattice | Holds state at program points |
| `DeadCodeAnalysis` | Pre-built | Detect live blocks/edges |
| `IntegerRangeAnalysis` | Pre-built | Infer integer ranges |
| `FlatAffineValueConstraints` | Affine analysis | Model loop domains |
| `MemRefDependenceGraph` | Affine analysis | Track memory dependences |
