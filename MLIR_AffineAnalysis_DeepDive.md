# MLIR Affine Analysis - Deep Dive & Code Examples

## Overview
The affine analysis subsystem in MLIR provides tools for reasoning about affine loop nests, memory access patterns, and iteration-dependent expressions using polyhedral models.

---

## 1. FlatAffineValueConstraints API Details

**Location**: `mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h`

### 1.1 Constructor & Basic Operation

```cpp
// Create empty constraint system with specified dimensions and symbols
FlatAffineValueConstraints cst(numDimVars, numSymbolVars, numLocalVars);

// Create from another constraint system
FlatAffineValueConstraints cst(other);

// From an AffineMap
FlatAffineValueConstraints cst(affineMap);
```

### 1.2 Adding Loop Bounds

```cpp
// Add constraints from affine.for operation
LogicalResult addAffineForOpDomain(AffineForOp forOp) {
  // Adds constraints for loop induction variable
  // Lower bound: i >= lb (extracted from affineForOp.lowerBound())
  // Upper bound: i < ub (extracted from affineForOp.upperBound())
  // Operands from bounds are added as symbols
}

// Add constraints from affine.parallel
LogicalResult addAffineParallelOpDomain(AffineParallelOp parallelOp);

// Add constraints from multiple loop bounds at once
LogicalResult addDomainFromSliceMaps(
    ArrayRef<AffineMap> lbMaps,     // Lower bound maps
    ArrayRef<AffineMap> ubMaps,     // Upper bound maps
    ArrayRef<Value> operands);      // Common operands for all maps
```

### 1.3 Adding If Conditions

```cpp
// Add constraints from affine.if condition (IntegerSet)
void addAffineIfOpDomain(AffineIfOp ifOp) {
  // Extracts linear constraints from IntegerSet
  // Example: if (i >= j & i+j < 10) creates:
  //   i - j >= 0
  //   i + j < 10
}
```

### 1.4 Adding Arbitrary Bounds

```cpp
// Add bound for variable at position `pos`
// BoundType: LB (lower bound), UB (upper bound), EQ (equality)
LogicalResult addBound(
    presburger::BoundType type,     // LB, UB, or EQ
    unsigned pos,                   // Variable position
    AffineMap boundMap,             // Expression for bound
    ValueRange operands);           // Values referenced in map
```

### 1.5 Adding Induction Variables

```cpp
// Add loop IV or terminal symbol to constraint system
// Terminal symbol = loop IV or SSA value that isn't computed from other IVs
LogicalResult addInductionVarOrTerminalSymbol(Value val);

// Convert all loop IV symbols to dimension variables
void convertLoopIVSymbolsToDims();
```

### 1.6 Querying Bounds

```cpp
// Extract inequality as AffineValueMap (affine map + operands)
void getIneqAsAffineValueMap(
    unsigned pos,           // Variable position
    unsigned ineqPos,       // Inequality constraint position
    AffineValueMap &vmap,   // Output: map + operands
    MLIRContext *context);

// Returns lowerbound or upper bound depending on sign
// Example: If constraint is "2*i + j <= 10", returns map as:
//   5 - i/2 if pos=j (upper bound)
//   or relationship in terms of pos
```

### 1.7 Affine Map Composition

```cpp
// Compose affine value map with constraint system
// Results of map become new dimension variables
// Example: if cst has dimensions [i, j] and vMap is "f(x,y)",
// adds dimensions [f(x,y), i, j]
LogicalResult composeMap(const AffineValueMap &vMap);
```

---

## 2. FlatAffineRelation - Domain/Range Separation

**Location**: `mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h` (Lines 167+)

```cpp
class FlatAffineRelation : public FlatAffineValueConstraints {
public:
  // Constructor: domain variables, range variables, symbol variables, local vars
  FlatAffineRelation(unsigned numDomainDims, unsigned numRangeDims,
                     unsigned numSymbols, unsigned numLocals);

  // Get domain and range as separate constraint systems
  FlatAffineValueConstraints getDomainSet() const;
  FlatAffineValueConstraints getRangeSet() const;

  // Dimensions:
  // [0, numDomainDims) = domain (input variables)
  // [numDomainDims, numDomainDims + numRangeDims) = range (output variables)
};
```

**Use Cases**:
- Modeling access functions: domain=iteration variables, range=memory location
- Dependence relations: domain=source iteration, range=sink iteration
- Function mappings: domain=inputs, range=outputs

**Example: Modeling memref access**:
```cpp
// Access: A[2*i + j] where i,j are loop IVs
// Domain dims: [i, j] - iteration space
// Range dims: [index] - memory indices
FlatAffineRelation access(2, 1, 0, 0);  // 2 domain, 1 range, no symbols
// Add constraint: index = 2*i + j (equality)
access.addConstraint(/*index*/ index = 2*i + j);
```

---

## 3. Iteration-Dependent Expression Handling

### 3.1 What FlatAffineValueConstraints Cannot Directly Handle

The system works with **affine functions** only:
```
Valid:   f(i, j, s) = 2*i + 3*j + s + 5
Invalid: f(i, j) = i*j  (non-linear)
Invalid: f(i) = 2^i     (non-polynomial)
```

### 3.2 Modeling Iteration-Dependent Expressions

```cpp
// Example 1: Loop-dependent bound
// for i = c0 to c1 {
//   for j = c0 to i {  // j depends on i
//     ...
//   }
// }

FlatAffineValueConstraints cst;
// Add outer loop bounds
cst.addAffineForOpDomain(outerForOp);  // Creates i with bounds c0 to c1

// Add inner loop bounds that reference i
cst.addAffineForOpDomain(innerForOp);  // Creates j with bounds c0 to i
// This automatically adds: j >= c0, j <= i (linear constraint!)

// Query: what is maximum value of j?
// Answer: min(outerBound, i) at each iteration
```

### 3.3 Parametric Iteration Spaces

```cpp
// Example: Loop with symbolic upper bound
// %ub = ... (unknown value)
// for i = 0 to %ub {
//   ...
// }

FlatAffineValueConstraints cst;
// Add %ub as a symbol (not a dimension)
cst.addBound(presburger::BoundType::UB, 0, affineMap, {ub});
// Now system knows: 0 <= i < %ub
// But %ub is symbolic - exact value unknown at compile time
// Can still answer: "Is i = 5 always valid?" (depends on %ub)
```

---

## 4. MemRefDependenceGraph

**Location**: `mlir/include/mlir/Dialect/Affine/Analysis/Utils.h` (Lines 50-250)

### 4.1 Graph Structure

```cpp
struct MemRefDependenceGraph {
  struct Node {
    unsigned id;              // Unique node ID
    Operation *op;            // Top-level statement or loop nest

    SmallVector<Operation *, 4> loads;
    SmallVector<Operation *, 4> stores;
    SmallVector<Operation *, 4> memrefLoads;   // Non-affine loads
    SmallVector<Operation *, 4> memrefStores;  // Non-affine stores
    SmallVector<Operation *, 4> memrefFrees;

    DenseSet<Value> privateMemrefs;  // Memrefs only used in this node
  };

  struct Edge {
    unsigned id;
    unsigned src, dst;
    Value value;              // SSA value creating dependence

    // Type of dependence:
    // - memref: memory location shared between ops
    // - SSA: SSA value dependence
  };

  Block &block;              // Block containing the nodes
};
```

### 4.2 Query Operations

```cpp
// Connectivity queries
bool hasDependencePath(unsigned srcId, unsigned dstId) const;
  // True if there's a path in dependence graph from src to dst

unsigned getIncomingMemRefAccesses(unsigned id, Value memref) const;
  // Count of store operations to memref that write to data read by id

unsigned getOutEdgeCount(unsigned id, Value memref = nullptr) const;
  // Count of outgoing memref edges from node id

void gatherDefiningNodes(unsigned id, DenseSet<unsigned> &definingNodes) const;
  // Find all nodes that define SSA values used in id

// Loop fusion utilities
Operation *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId) const;
  // Find where to insert fused loop while preserving dependences

void updateEdges(unsigned srcId, unsigned dstId,
                 const DenseSet<Value> &privateMemRefs,
                 bool removeSrcId);
  // Update graph after fusion
```

### 4.3 Edge Iteration

```cpp
// Iterate over memory dependence edges
graph.forEachMemRefInputEdge(nodeId, [&](const MemRefDependenceGraph::Edge &e) {
  // e.src -> nodeId carries memref dependence
  Operation *sourceOp = graph.nodes[e.src].op;
  Value memref = e.value;
});

graph.forEachMemRefOutputEdge(nodeId, [&](const MemRefDependenceGraph::Edge &e) {
  // nodeId -> e.dst carries memref dependence
  Operation *destOp = graph.nodes[e.dst].op;
});
```

---

## 5. Affine Loop Utilities

**Location**: `mlir/include/mlir/Dialect/Affine/Analysis/Utils.h` (Lines 270-430)

### 5.1 Loop Navigation

```cpp
/// Get affine.for loop IVs from outermost to innermost
void getAffineForIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops);

/// Get affine.for and affine.parallel IVs (stops at loop boundaries)
void getAffineIVs(Operation &op, SmallVectorImpl<Value> &ivs);

/// Get enclosing affine operations (for, parallel, if)
void getEnclosingAffineOps(Operation &op, SmallVectorImpl<Operation *> *ops);

/// Nesting depth = number of surrounding loops
unsigned getNestingDepth(Operation *op);

/// Example: Finding loop context
SmallVector<AffineForOp, 4> enclosingLoops;
getAffineForIVs(*affineLoad, &enclosingLoops);
// Iterate bounds of each loop
for (AffineForOp loop : enclosingLoops) {
  AffineMap lowerBound = loop.getlowerBound();
  AffineMap upperBound = loop.getUpperBound();
  ValueRange operands = loop.getLowerBoundOperands();
}
```

### 5.2 Loop Properties

```cpp
/// Check if loop is parallel and contains reduction
bool isLoopParallelAndContainsReduction(AffineForOp forOp);

/// Get sequential loops in a loop nest
void getSequentialLoops(AffineForOp forOp,
                        llvm::SmallDenseSet<Value, 8> *sequentialLoops);
```

---

## 6. Complete Example: State Access Analysis

Here's how to build a state access analyzer using these tools:

```cpp
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

// State access pattern representation
struct LoopStateAccessPattern {
  // Which iteration variables affect state?
  SmallVector<Value, 4> affectedLoopIVs;

  // Constraint system describing state access ranges
  affine::FlatAffineValueConstraints accessConstraints;

  // Is access uniform across iterations?
  bool isIterationIndependent = false;

  static LoopStateAccessPattern join(const LoopStateAccessPattern &lhs,
                                     const LoopStateAccessPattern &rhs) {
    // Merge two access patterns
    // Union affected loop IVs
    // Join constraint systems (union of valid iteration points)
    LoopStateAccessPattern result;
    result.isIterationIndependent =
        lhs.isIterationIndependent && rhs.isIterationIndependent;
    return result;
  }

  bool operator==(const LoopStateAccessPattern &other) const {
    return accessConstraints == other.accessConstraints &&
           isIterationIndependent == other.isIterationIndependent;
  }
};

// Custom sparse analysis
class LoopStateAccessAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
        dataflow::Lattice<LoopStateAccessPattern>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::Lattice<LoopStateAccessPattern> *> operands,
      ArrayRef<dataflow::Lattice<LoopStateAccessPattern> *> results) override {

    // For each result value
    for (auto [i, result] : llvm::enumerate(results)) {
      LoopStateAccessPattern pattern;

      // Get enclosing affine loops
      SmallVector<AffineForOp, 4> loops;
      affine::getAffineForIVs(*op, &loops);

      for (AffineForOp loop : loops) {
        pattern.affectedLoopIVs.push_back(loop.getInductionVar());

        // Add loop bounds to constraint system
        pattern.accessConstraints.addAffineForOpDomain(loop);
      }

      // Check if operation accesses state uniformly
      if (loops.empty()) {
        pattern.isIterationIndependent = true;
      }

      // Join with operand patterns
      for (const auto *operandLattice : operands) {
        if (operandLattice) {
          result->join(operandLattice->getValue());
        }
      }

      result->join(pattern);
    }

    return success();
  }

  void setToEntryState(
      dataflow::Lattice<LoopStateAccessPattern> *lattice) override {
    // Entry: no loops, no constraints
    propagateIfChanged(lattice, lattice->join(LoopStateAccessPattern{}));
  }
};

// Usage in a pass
class LoopStateAccessPass : public PassWrapper<...> {
  void runOnOperation() override {
    Operation *module = getOperation();

    dataflow::DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<LoopStateAccessAnalysis>();

    if (failed(solver.initializeAndRun(module))) {
      return signalPassFailure();
    }

    // Query results
    module->walk([&](mlir::Value val) {
      const auto *lattice =
          solver.lookupState<dataflow::Lattice<LoopStateAccessPattern>>(val);

      if (lattice && !lattice->getValue().isUninitialized()) {
        const auto &pattern = lattice->getValue().getValue();

        llvm::outs() << "Value: " << val << "\n";
        llvm::outs() << "  Affected loops: "
                     << pattern.affectedLoopIVs.size() << "\n";
        llvm::outs() << "  Iteration-independent: "
                     << pattern.isIterationIndependent << "\n";
      }
    });
  }
};
```

---

## 7. Presburger Library Integration

The constraint system uses MLIR's Presburger library for linear integer arithmetic.

```cpp
// Related headers
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"

// FlatAffineValueConstraints extends FlatLinearValueConstraints
// which uses Presburger::IntegerRelation underneath

// Capabilities:
// - Linear constraints: ax + by + c <= 0
// - Equality constraints: ax + by + c == 0
// - Rational arithmetic (constraints can have rational coefficients)
// - Fourier-Motzkin elimination
// - Lexicographic ordering
// - Quantifier elimination (exists/forall)
```

---

## 8. Common Patterns & Recipes

### 8.1 Extract Iteration Bounds for a Loop

```cpp
void analyzeLoopBounds(AffineForOp forOp) {
  // Get lower/upper bound maps
  AffineMap lowerBound = forOp.getLowerBound();
  AffineMap upperBound = forOp.getUpperBound();

  ValueRange operands = forOp.getLowerBoundOperands();
  Value iv = forOp.getInductionVar();

  // Create constraint system
  affine::FlatAffineValueConstraints cst;
  cst.addAffineForOpDomain(forOp);

  // Now cst contains: iv >= lowerBound(operands) && iv < upperBound(operands)

  // Get as affine value map
  affine::AffineValueMap ivBounds;
  cst.getIneqAsAffineValueMap(/*iv position*/ 0, /*constraint*/ 0,
                               ivBounds, forOp->getContext());
}
```

### 8.2 Check If Two Iterations Can Be Parallelized

```cpp
bool canParallelize(AffineForOp loop1, AffineForOp loop2) {
  if (affine::isLoopParallelAndContainsReduction(loop1) ||
      affine::isLoopParallelAndContainsReduction(loop2)) {
    return false;
  }

  // Check dependence graph
  // (More complex dependence analysis)
  return true;
}
```

### 8.3 Extract All Constrained Iteration Points

```cpp
void getConstrainedIterations(affine::AffineForOp loop) {
  affine::FlatAffineValueConstraints cst;
  cst.addAffineForOpDomain(loop);

  // Get integer points in the polytope
  // (Using Presburger library underneath)
  // Can enumerate all feasible iterations
}
```

---

## 9. Typical Issues & Solutions

| Issue | Solution |
|-------|----------|
| Semi-affine expressions (e.g., min/max) | Use lower/upper bounds explicitly; model as multiple affine constraints |
| Non-linear loop bounds | Can't directly support; linearize if possible or use symbolic approaches |
| Variable-dependent iterations | Use symbols for unknown values; add constraints from if-conditions |
| Memref aliasing | Use AliasAnalysis or MemRefDependenceGraph |
| Complex dependences | Use FlatAffineRelation with source/sink iteration variables |

---

## References

All paths relative to `llvm-project/mlir/`:

- Core: `include/mlir/Dialect/Affine/Analysis/AffineStructures.h`
- Utils: `include/mlir/Dialect/Affine/Analysis/Utils.h`
- IR: `include/mlir/IR/AffineExpr.h`, `include/mlir/IR/AffineMap.h`
- Presburger: `include/mlir/Analysis/Presburger/`
