# MLIR Dataflow Analysis Infrastructure - Comprehensive Research Report

## Overview
MLIR provides a sophisticated, modular dataflow analysis framework designed for writing various types of analyses (forward/backward, sparse/dense) that operate on SSA values and program state. The framework is designed to be composable, allowing multiple analyses to cooperate through a unified solver infrastructure.

---

## Part 1: Core Dataflow Framework

### 1.1 Main Architecture Components

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlowFramework.h](llvm-project/mlir/include/mlir/Analysis/DataFlowFramework.h)

#### DataFlowSolver
The orchestrator class that manages all analyses and implements the fixed-point iteration algorithm.

**Key Methods**:
```cpp
class DataFlowSolver {
public:
  // Load an analysis into the solver
  template <typename AnalysisT, typename... Args>
  AnalysisT *load(Args &&...args);

  // Initialize and run the analysis
  LogicalResult initializeAndRun(Operation *top);

  // Query analysis results
  template <typename StateT, typename AnchorT>
  const StateT *lookupState(AnchorT anchor) const;

  // Erase analysis states when IR changes
  template <typename AnchorT>
  void eraseState(AnchorT anchor);
  void eraseAllStates();

  // Get program points for insertion
  ProgramPoint *getProgramPointBefore(Operation *op);
  ProgramPoint *getProgramPointAfter(Operation *op);
  ProgramPoint *getProgramPointBefore(Block *block);
  ProgramPoint *getProgramPointAfter(Block *block);
};
```

**Configuration**:
```cpp
class DataFlowConfig {
public:
  DataFlowConfig &setInterprocedural(bool enable);
  bool isInterprocedural() const;
};
```

**Usage Pattern**:
```cpp
DataFlowSolver solver;
solver.load<DeadCodeAnalysis>();
solver.load<IntegerRangeAnalysis>();
if (failed(solver.initializeAndRun(operation)))
  return signalPassFailure();

// Query results
auto *state = solver.lookupState<IntegerValueRangeLattice>(value);
```

#### DataFlowAnalysis (Base Class)
[Lines 589-680 in DataFlowFramework.h]

Abstract base class for all analyses to implement.

```cpp
class DataFlowAnalysis {
public:
  virtual ~DataFlowAnalysis();

  // Subclasses must implement:
  virtual LogicalResult initialize(Operation *top) = 0;
  virtual LogicalResult visit(ProgramPoint *point) = 0;
  virtual void initializeEquivalentLatticeAnchor(Operation *top) {}

protected:
  // Register and query custom lattice anchors
  template <typename AnchorT>
  void registerAnchorKind();

  template <typename AnchorT, typename... Args>
  AnchorT *getLatticeAnchor(Args &&...args);

  // Dependency and state management
  void addDependency(AnalysisState *state, ProgramPoint *point);
  void propagateIfChanged(AnalysisState *state, ChangeResult changed);
};
```

#### AnalysisState (Base Class)
```cpp
class AnalysisState {
public:
  LatticeAnchor getAnchor() const;
  virtual void print(raw_ostream &os) const = 0;

protected:
  // Called when state is updated - can enqueue dependent work
  virtual void onUpdate(DataFlowSolver *solver) const;
  void addDependency(ProgramPoint *point, DataFlowAnalysis *analysis);
};
```

### 1.2 ChangeResult & Basic Types

```cpp
enum class [[nodiscard]] ChangeResult {
  NoChange,
  Change,
};
```

Used to indicate whether analysis states changed during updates. Critical for fixpoint convergence.

### 1.3 Program Points & Lattice Anchors

**Program Points**: Represent specific locations in IR execution
- Before/after operations
- At block entries/exits

**Lattice Anchors**: The targets of analysis (can be custom)
- Values (for sparse analyses tracking SSA values)
- Operations/Blocks (for dense analyses)
- Custom generic anchors

---

## Part 2: Specific Analysis Frameworks

### 2.1 Sparse Dataflow Analysis

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h)

Sparse analysis propagates information only through **explicitly used values** (SSA use-def chains).

#### AbstractSparseLattice

```cpp
class AbstractSparseLattice : public AnalysisState {
public:
  AbstractSparseLattice(Value value) : AnalysisState(value) {}

  Value getAnchor() const { return cast<Value>(AnalysisState::getAnchor()); }

  // Join/Meet operations for lattice elements
  virtual ChangeResult join(const AbstractSparseLattice &rhs) {
    return ChangeResult::NoChange;
  }

  virtual ChangeResult meet(const AbstractSparseLattice &rhs) {
    return ChangeResult::NoChange;
  }
};

// Template version with specific lattice value type
template <typename ValueT>
class Lattice : public AbstractSparseLattice {
public:
  ValueT &getValue();
  ChangeResult join(const ValueT &rhs);
  ChangeResult meet(const ValueT &rhs);
};
```

**Requirements for ValueT**:
- `static ValueT join(const ValueT &lhs, const ValueT &rhs);`
- `bool operator==(const ValueT &rhs) const;`
- Optional: `static ValueT meet(const ValueT &lhs, const ValueT &rhs);`

#### Sparse Forward Analysis

```cpp
class AbstractSparseForwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint *point) override;
};

// Template for concrete implementations
template <typename StateT>
class SparseForwardDataFlowAnalysis : public AbstractSparseForwardDataFlowAnalysis {
public:
  // Must override:
  virtual LogicalResult visitOperation(Operation *op,
                                       ArrayRef<const StateT *> operands,
                                       ArrayRef<StateT *> results) = 0;

  virtual void visitExternalCall(CallOpInterface call,
                                 ArrayRef<const StateT *> argumentLattices,
                                 ArrayRef<StateT *> resultLattices);

  virtual void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<StateT *> nonSuccessorInputLattices);
};
```

#### Sparse Backward Analysis

```cpp
class AbstractSparseBackwardDataFlowAnalysis : public DataFlowAnalysis;

template <typename StateT>
class SparseBackwardDataFlowAnalysis : public AbstractSparseBackwardDataFlowAnalysis {
public:
  virtual LogicalResult visitOperation(Operation *op,
                                       ArrayRef<StateT *> operands,
                                       ArrayRef<const StateT *> results) = 0;

  virtual void visitBranchOperand(OpOperand &operand);

  virtual void visitCallOperand(OpOperand &operand);

  virtual void setToExitState(StateT *lattice) = 0;

  virtual void visitNonControlFlowArguments(
      RegionSuccessor &successor,
      ArrayRef<BlockArgument> arguments);
};
```

### 2.2 Dense Dataflow Analysis

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h)

Dense analysis attaches lattice information to **every program point** (operations and blocks).

#### AbstractDenseLattice
```cpp
class AbstractDenseLattice : public AnalysisState {
public:
  virtual ChangeResult join(const AbstractDenseLattice &rhs);
  virtual ChangeResult meet(const AbstractDenseLattice &rhs);
};

// Template version
template <typename ValueT>
class DenseLattice : public AbstractDenseLattice;
```

#### Dense Forward Analysis
```cpp
class AbstractDenseForwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint *point) override;

protected:
  virtual LogicalResult visitOperationImpl(Operation *op,
                                          const AbstractDenseLattice &before,
                                          AbstractDenseLattice *after) = 0;

  virtual AbstractDenseLattice *getLattice(LatticeAnchor anchor) = 0;

  virtual const AbstractDenseLattice *getLatticeFor(ProgramPoint *dependent,
                                                    LatticeAnchor anchor) = 0;

  virtual void setToEntryState(AbstractDenseLattice *lattice) = 0;
};

// Concrete template
template <typename StateT>
class DenseForwardDataFlowAnalysis : public AbstractDenseForwardDataFlowAnalysis;
```

#### Dense Backward Analysis
Similar structure with `AbstractDenseBackwardDataFlowAnalysis` and `DenseBackwardDataFlowAnalysis` template.

---

## Part 3: Pre-built Analyses

### 3.1 Dead Code Analysis

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h)

Determines which blocks, operations, and control-flow edges are live.

```cpp
class Executable : public AnalysisState {
public:
  ChangeResult setToLive();
  bool isLive() const;
  void blockContentSubscribe(DataFlowAnalysis *analysis);
};

class PredecessorState : public AnalysisState {
  // Track control-flow predecessors
};

class DeadCodeAnalysis : public DataFlowAnalysis;
```

**Purpose**: Essential prerequisite for other analyses that need to ignore dead code.

### 3.2 Liveness Analysis (Sparse, Backward)

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/LivenessAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/LivenessAnalysis.h)

Determines which SSA values are "live" (used to compute results or memory effects).

```cpp
struct Liveness : public AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Liveness)

  bool isLive = false;

  ChangeResult markLive();
  ChangeResult meet(const AbstractSparseLattice &other) override;
  void print(raw_ostream &os) const override;
};

class LivenessAnalysis : public SparseBackwardDataFlowAnalysis<Liveness> {
public:
  LogicalResult visitOperation(Operation *op, ArrayRef<Liveness *> operands,
                               ArrayRef<const Liveness *> results) override;

  void visitBranchOperand(OpOperand &operand) override;
  void visitCallOperand(OpOperand &operand) override;
  void setToExitState(Liveness *lattice) override;
};
```

**Definition**: A value is "live" iff:
1. It has memory effects, OR
2. It's returned by a public function, OR
3. It's used to compute a value of type (1) or (2)

**Usage**:
```cpp
struct RunLivenessAnalysis {
  // Runs liveness analysis on the IR
};
```

### 3.3 Constant Propagation Analysis (Sparse, Forward)

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h)

Determines constant-valued results by speculatively folding operations.

```cpp
class ConstantValue {
public:
  Attribute getConstantValue() const;
  Dialect *getConstantDialect() const;

  bool isUninitialized() const;

  static ConstantValue getUnknownConstant();
  static ConstantValue join(const ConstantValue &lhs,
                            const ConstantValue &rhs);
};

class SparseConstantPropagation
    : public SparseForwardDataFlowAnalysis<Lattice<ConstantValue>> {
public:
  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const Lattice<ConstantValue> *> operands,
                               ArrayRef<Lattice<ConstantValue> *> results) override;

  void setToEntryState(Lattice<ConstantValue> *lattice) override;
};
```

**Principle**: Sparse Conditional Constant Propagation (SCCP) when combined with dead-code analysis.

### 3.4 Integer Range Analysis (Sparse, Forward)

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h)

Infers integer value ranges of SSA values using operations with `InferIntRangeInterface`.

```cpp
class IntegerValueRangeLattice : public Lattice<IntegerValueRange> {
public:
  void onUpdate(DataFlowSolver *solver) const override;
  // Automatically updates constant values when range narrows to single value
};

class IntegerRangeAnalysis
    : public SparseForwardDataFlowAnalysis<IntegerValueRangeLattice> {
public:
  void setToEntryState(IntegerValueRangeLattice *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const IntegerValueRangeLattice *> operands,
                               ArrayRef<IntegerValueRangeLattice *> results) override;

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<IntegerValueRangeLattice *> nonSuccessorInputLattices) override;
};

// Helper functions
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Operation *op);
LogicalResult staticallyNonNegative(DataFlowSolver &solver, Value v);
LogicalResult maybeReplaceWithConstant(DataFlowSolver &solver,
                                       RewriterBase &rewriter, Value value);
```

**Dependency**: Depends on `DeadCodeAnalysis` (silent no-op if not loaded).

**Interface**: Operations can implement `InferIntRangeInterface` to provide custom range semantics.

### 3.5 Strided Metadata Range Analysis (Sparse, Forward)

**File**: [llvm-project/mlir/include/mlir/Analysis/DataFlow/StridedMetadataRangeAnalysis.h](llvm-project/mlir/include/mlir/Analysis/DataFlow/StridedMetadataRangeAnalysis.h)

Analyzes strided metadata information for operations implementing `InferStridedMetadataInterface`.

---

## Part 4: Integration Points & Composition

### 4.1 How Analyses Compose

**Multiple Analyses in One Solver**:

```cpp
DataFlowSolver solver;
// Load analyses - they share the same solver state space
solver.load<DeadCodeAnalysis>();
solver.load<ConstantPropagationAnalysis>();
solver.load<IntegerRangeAnalysis>();

if (failed(solver.initializeAndRun(operation)))
  return failure();

// All analyses can access each other's results through the solver
auto *constantState = solver.lookupState<Lattice<ConstantValue>>(value);
auto *rangeState = solver.lookupState<IntegerValueRangeLattice>(value);
```

### 4.2 Analysis Dependencies

**Explicit Dependencies** (via solver.load order):
- `IntegerRangeAnalysis` depends on `DeadCodeAnalysis` (checks dynamically)
- Analyses can query other analyses' states via `solver.lookupState()`

**Implicit Dependencies** (via state updates):
- When one analysis updates a state, dependent analyses are re-invoked
- `useDefSubscribe()` in sparse analyses for efficient use-def chain tracking

### 4.3 Real-world Composition Example

**File**: [llvm-project/mlir/lib/Dialect/Arith/Transforms/UnsignedWhenEquivalent.cpp](llvm-project/mlir/lib/Dialect/Arith/Transforms/UnsignedWhenEquivalent.cpp) (Lines 33-145)

```cpp
// Create solver with multiple analyses
DataFlowSolver solver;
solver.load<DeadCodeAnalysis>();
solver.load<IntegerRangeAnalysis>();
if (failed(solver.initializeAndRun(op)))
  return signalPassFailure();

// Query results from multiple analyses
static LogicalResult isCmpIConvertable(DataFlowSolver &solver, CmpIOp op) {
  CmpIPredicate pred = op.getPredicate();
  switch (pred) {
  case CmpIPredicate::sle:
  case CmpIPredicate::slt:
  case CmpIPredicate::sge:
  case CmpIPredicate::sgt:
    return success(llvm::all_of(op.getOperands(), [&solver](Value v) -> bool {
      // Uses IntegerRangeAnalysis indirectly via helper
      return succeeded(staticallyNonNegative(solver, v));
    }));
  default:
    return failure();
  }
}

// Listener pattern for keeping analyses synchronized with rewrites
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}
protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults())
      s.eraseState(res);
  }
  DataFlowSolver &s;
};

// Apply patterns
DataFlowListener listener(solver);
walkAndApplyPatterns(op, std::move(patterns), &listener);
```

### 4.4 Another Real-world Example: Integer Range Optimizations

**File**: [llvm-project/mlir/lib/Dialect/Arith/Transforms/IntRangeOptimizations.cpp](llvm-project/mlir/lib/Dialect/Arith/Transforms/IntRangeOptimizations.cpp) (Lines 1-150)

Shows pattern rewriting that uses the solver to materialize constants:

```cpp
// Utility functions for querying analysis results
static std::optional<APInt> getMaybeConstantValue(DataFlowSolver &solver,
                                                  Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
    return std::nullopt;
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  return inferredRange.getConstantValue();
}

static void copyIntegerRange(DataFlowSolver &solver, Value oldVal,
                             Value newVal) {
  auto *oldState = solver.lookupState<IntegerValueRangeLattice>(oldVal);
  if (!oldState)
    return;
  (void)solver.getOrCreateState<IntegerValueRangeLattice>(newVal)->join(*oldState);
}

// Pattern to materialize constants discovered by analysis
struct MaterializeKnownConstantValues : public RewritePattern {
  MaterializeKnownConstantValues(MLIRContext *context, DataFlowSolver &s)
      : RewritePattern::RewritePattern(Pattern::MatchAnyOpTypeTag(),
                                       /*benefit=*/1, context),
        solver(s) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto needsReplacing = [&](Value v) {
      return getMaybeConstantValue(solver, v).has_value() && !v.use_empty();
    };
    bool hasConstantResults = llvm::any_of(op->getResults(), needsReplacing);
    // ... perform transformation
  }
};
```

---

## Part 5: Affine Analysis & FlatAffineValueConstraints

### 5.1 FlatAffineValueConstraints

**File**: [llvm-project/mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h](llvm-project/mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h)

Extends `FlatLinearValueConstraints` with affine-specific utilities for representing affine loop domains and iteration-dependent expressions.

```cpp
class FlatAffineValueConstraints : public FlatLinearValueConstraints {
public:
  // Affine for-loop bounds
  LogicalResult addAffineForOpDomain(AffineForOp forOp);

  // Parallel loop handling
  LogicalResult addAffineParallelOpDomain(AffineParallelOp parallelOp);

  // If condition constraints
  void addAffineIfOpDomain(AffineIfOp ifOp);

  // Slice bounds from maps
  LogicalResult addDomainFromSliceMaps(ArrayRef<AffineMap> lbMaps,
                                       ArrayRef<AffineMap> ubMaps,
                                       ArrayRef<Value> operands);

  // Bounds from AffineMap
  LogicalResult addBound(presburger::BoundType type, unsigned pos,
                         AffineMap boundMap, ValueRange operands);

  // Add induction variables and symbols
  LogicalResult addInductionVarOrTerminalSymbol(Value val);

  // Slice operations
  LogicalResult addSliceBounds(ArrayRef<Value> values,
                               ArrayRef<AffineMap> lbMaps,
                               ArrayRef<AffineMap> ubMaps,
                               ArrayRef<Value> operands);

  // Transform symbols to dimensions
  void convertLoopIVSymbolsToDims();

  // Extract bounds as affine value maps
  void getIneqAsAffineValueMap(unsigned pos, unsigned ineqPos,
                               AffineValueMap &vmap,
                               MLIRContext *context) const;
};
```

### 5.2 FlatAffineRelation

```cpp
class FlatAffineRelation : public FlatAffineValueConstraints {
public:
  unsigned getNumDomainDims() const;
  unsigned getNumRangeDims() const;

  FlatAffineValueConstraints getDomainSet() const;
  FlatAffineValueConstraints getRangeSet() const;
};
```

Represents affine relations with separated domain and range dimensions - useful for dependence analysis.

### 5.3 Affine Analysis Utilities

**File**: [llvm-project/mlir/include/mlir/Dialect/Affine/Analysis/Utils.h](llvm-project/mlir/include/mlir/Dialect/Affine/Analysis/Utils.h) (Lines 300-350)

```cpp
// Memory dependence graph for affine programs
struct MemRefDependenceGraph {
  struct Node {
    unsigned id;
    Operation *op;
    SmallVector<Operation *, 4> loads;
    SmallVector<Operation *, 4> stores;
    SmallVector<Operation *, 4> memrefLoads;
    SmallVector<Operation *, 4> memrefStores;
    SmallVector<Operation *, 4> memrefFrees;
    DenseSet<Value> privateMemrefs;
  };

  struct Edge {
    unsigned id;
    unsigned src, dst;
    Value value;
  };

  // Query operations
  bool hasDependencePath(unsigned srcId, unsigned dstId) const;
  unsigned getIncomingMemRefAccesses(unsigned id, Value memref) const;
  unsigned getOutEdgeCount(unsigned id, Value memref = nullptr) const;
  void gatherDefiningNodes(unsigned id, DenseSet<unsigned> &definingNodes) const;
};

// Loop utilities
void getAffineForIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops);
void getAffineIVs(Operation &op, SmallVectorImpl<Value> &ivs);
void getEnclosingAffineOps(Operation &op, SmallVectorImpl<Operation *> *ops);
unsigned getNestingDepth(Operation *op);
```

### 5.4 Capabilities for Iteration-Dependent Expressions

**FlatAffineValueConstraints** handles:

1. **Loop bounds**: Models affine.for upper/lower bounds
2. **Symbolic values**: Non-loop-dependent values as symbols
3. **Inequality constraints**: Linear constraints on dimensions/symbols
4. **Affine maps composition**: Can represent complex affine expressions
5. **Presburger arithmetic**: Underlying representation for rational linear arithmetic

**Example: Adding loop bounds**:
```cpp
FlatAffineValueConstraints constraints;

// Add loop bounds for affine.for %i = %lb to %ub
// This creates constraints like: i >= lb, i < ub
constraints.addAffineForOpDomain(forOp);

// Can now query feasible iteration ranges
// Check if specific iteration points are feasible
```

---

## Part 6: Building Custom Analyses

### 6.1 Template for Sparse Forward Analysis

```cpp
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

// 1. Define lattice value type
struct MyLatticeValue {
  int state = 0;

  // Required methods
  static MyLatticeValue join(const MyLatticeValue &lhs,
                             const MyLatticeValue &rhs) {
    return MyLatticeValue{std::max(lhs.state, rhs.state)};
  }

  bool operator==(const MyLatticeValue &rhs) const {
    return state == rhs.state;
  }

  void print(raw_ostream &os) const { os << state; }
};

// 2. Implement analysis
class MyAnalysis : public SparseForwardDataFlowAnalysis<Lattice<MyLatticeValue>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const Lattice<MyLatticeValue> *> operands,
                               ArrayRef<Lattice<MyLatticeValue> *> results) override {
    // Implement transfer function
    return success();
  }

  void setToEntryState(Lattice<MyLatticeValue> *lattice) override {
    lattice->join(MyLatticeValue{0});
  }
};

// 3. Use in pass
DataFlowSolver solver;
solver.load<MyAnalysis>();
if (failed(solver.initializeAndRun(op)))
  return failure();

const auto *state = solver.lookupState<Lattice<MyLatticeValue>>(value);
```

### 6.2 Template for Dense Forward Analysis

```cpp
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"

// 1. Define lattice
template <typename ValueT>
class DenseForwardAnalysis
    : public DenseForwardDataFlowAnalysis<DenseLattice<ValueT>> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

protected:
  LogicalResult visitOperationImpl(Operation *op,
                                  const DenseLattice<ValueT> &before,
                                  DenseLattice<ValueT> *after) override {
    // Transfer function from before to after
    return success();
  }

  DenseLattice<ValueT> *getLattice(LatticeAnchor anchor) override {
    return getOrCreate<DenseLattice<ValueT>>(anchor);
  }

  // ... other required methods
};
```

---

## Part 7: Key Design Patterns & Best Practices

### 7.1 Fixpoint Iteration Theory

The solver implements a classic fixpoint iteration:
1. Initialize analysis states
2. Process work items (program point + analysis pairs)
3. When state changes, propagate to dependents
4. Repeat until no more changes (fixpoint)

### 7.2 Dependency Management

**Implicit Dependencies** (automatic):
- In sparse analyses: use-def chains create dependencies
- `useDefSubscribe()` for efficient notifications

**Explicit Dependencies**:
- Created via `addDependency(state, point)` when analysis queries state

### 7.3 Lazy Initialization

Analyses don't need to know about all values upfront:
- Query uninitialized states returns null
- Solver "nudges" with default initialization if unresolved
- Allows analyses to focus on relevant computations

### 7.4 Interprocedural Analysis

```cpp
DataFlowConfig config;
config.setInterprocedural(true);  // Enter callees when available
config.setInterprocedural(false); // Skip callees (cheaper)

DataFlowSolver solver(config);
```

### 7.5 State Invalidation Pattern

When IR changes during transformation:
```cpp
class DataFlowListener : public RewriterBase::Listener {
protected:
  void notifyOperationErased(Operation *op) override {
    solver.eraseState(solver.getProgramPointAfter(op));
    for (Value res : op->getResults())
      solver.eraseState(res);
  }
};

// Apply rewrites while keeping solver in sync
walkAndApplyPatterns(op, patterns, &listener);
```

---

## Part 8: File Organization Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| **Core Framework** | `mlir/include/mlir/Analysis/DataFlowFramework.h` | `DataFlowSolver`, `DataFlowAnalysis`, `AnalysisState`, `ProgramPoint` |
| **Sparse Analysis** | `mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h` | `SparseForwardDataFlowAnalysis`, `SparseBackwardDataFlowAnalysis`, `AbstractSparseLattice` |
| **Dense Analysis** | `mlir/include/mlir/Analysis/DataFlow/DenseAnalysis.h` | `DenseForwardDataFlowAnalysis`, `DenseBackwardDataFlowAnalysis`, `AbstractDenseLattice` |
| **Dead Code** | `mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h` | Determines live blocks/edges |
| **Liveness** | `mlir/include/mlir/Analysis/DataFlow/LivenessAnalysis.h` | Value liveness tracking (backward sparse) |
| **Constant Prop** | `mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h` | SCCP (forward sparse) |
| **Integer Range** | `mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h` | Range inference (forward sparse) |
| **Strided Metadata** | `mlir/include/mlir/Analysis/DataFlow/StridedMetadataRangeAnalysis.h` | Metadata ranges (forward sparse) |
| **Affine Structures** | `mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h` | `FlatAffineValueConstraints`, loop domains |
| **Affine Utils** | `mlir/include/mlir/Dialect/Affine/Analysis/Utils.h` | Loop analysis, dependence graph |
| **Example Use** | `mlir/lib/Dialect/Arith/Transforms/UnsignedWhenEquivalent.cpp` | Real-world pattern with multiple analyses |
| **Example Use** | `mlir/lib/Dialect/Arith/Transforms/IntRangeOptimizations.cpp` | Pattern rewriting with solver support |

---

## Part 9: Query API Quick Reference

### Common Lookups

```cpp
// Lookup sparse lattice value
const IntegerValueRangeLattice *range =
  solver.lookupState<IntegerValueRangeLattice>(value);
if (range && !range->getValue().isUninitialized()) {
  APInt constant = range->getValue().getValue().getConstantValue();
}

// Lookup dense lattice
const MyDenseLattice *state =
  solver.lookupState<MyDenseLattice>(operation);

// Create or get state
MyLattice *state = solver.getOrCreateState<MyLattice>(value);

// Erase outdated state
solver.eraseState(value);
solver.eraseAllStates();
```

### Getting Program Points

```cpp
ProgramPoint *before = solver.getProgramPointBefore(op);
ProgramPoint *after = solver.getProgramPointAfter(op);
ProgramPoint *blockEntry = solver.getProgramPointBefore(block);
ProgramPoint *blockExit = solver.getProgramPointAfter(block);
```

---

## Recommendations for Your Use Case (Loop State Access Analysis)

Based on the LoopStateAccessAnalysis header in your project, here are relevant patterns:

1. **Use Dense Analysis** for tracking state at each program point
2. **Integrate with Affine Analysis** to handle affine loop domains
3. **Combine with IntegerRangeAnalysis** for iteration count inference
4. **Use FlatAffineValueConstraints** to model loop bounds and iteration spaces
5. **Query MemRefDependenceGraph** for alias/dependence information
6. **Consider backward analysis** if tracking "must reach" properties

---

## References

All file paths are relative to: `/Volumes/workplace/mlir-netsec/llvm-project/mlir/`

Key classes and their inheritance hierarchy:
- `DataFlowAnalysis` → `AbstractSparseForwardDataFlowAnalysis` → `SparseForwardDataFlowAnalysis<StateT>`
- `DataFlowAnalysis` → `AbstractSparseBackwardDataFlowAnalysis` → `SparseBackwardDataFlowAnalysis<StateT>`
- `DataFlowAnalysis` → `AbstractDenseForwardDataFlowAnalysis` → `DenseForwardDataFlowAnalysis<StateT>`
- `AnalysisState` → `AbstractSparseLattice` → `Lattice<ValueT>`
- `AnalysisState` → `AbstractDenseLattice` → `DenseLattice<ValueT>`
