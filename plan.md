# Plan: State-Independent Loop Parallelization (Analysis + Transformation)

**TL;DR:** Create two complementary components:

1. **Analysis Pass** (`StateAccessAnalysis`): Computes memory access patterns for each loop to determine which iterations access disjoint state regions. Results cached and reusable by other passes.
2. **Transformation Pass** (`StateIndependentParallelize`): Uses the analysis to convert `affine.for` loops to `affine.parallel` where iterations provably access non-overlapping state.

This generalizes beyond SCR patterns to any loop where iterations operate on disjoint state. Follows standard MLIR architecture (separate analysis from transformation) for modularity, reusability, and easier testing.

## Steps

### Part A: Analysis Pass

1. **Define LoopStateAccessAnalysis (Analysis Pass)**
   - New file: `src/transform/StateIndependentParallelization/LoopStateAccessAnalysis.h` and `.cpp`
   - Implement MLIR `AnalysisBase` pattern to compute state access information for each loop
   - Cache results for reuse by transformation and other passes

2. **Implement state access pattern computation**
   - Walk loop body and identify all memory/state access operations (memref.load, memref.store, spmc.*, etc.)
   - For each access:
     - Extract access region (affected memory elements)
     - Extract access pattern: which induction variables appear in indices?
     - Classify: constant region (loop-invariant), induction-dependent, or external-data-dependent
   - Build `LoopAccessInfo` structure storing:
     - Set of accessed memrefs/buffers
     - Access patterns per memref (region, dependency class)
     - Detected loop-carried dependencies
     - Iteration independence conclusion (bool)

3. **Implement conflict analysis**
   - Write `canIterationsConflict()` helper:
     - For two different iterations i, j: check if they access overlapping regions
     - Use FlatAffineAnalysis to compute access regions symbolically
     - Conservative: if regions cannot be proven disjoint, assume conflict
   - Write `hasLoopCarriedDependence()` helper:
     - Check if loop body has SSA data flow from one iteration output to next input
     - Identify which memrefs participate in dependencies

4. **Create Analysis output interface**
   - Provide `isLoopParallelizable(AffineForOp)` query method
   - Provide `getAccessInfo(AffineForOp)` to return detailed access information
   - Provide `getConflictingMemrefs(AffineForOp)` for diagnostics

### Part B: Transformation Pass

5. **Declare passes** in `src/transform/StateIndependentParallelization/Passes.td`
   - Declare `StateAccessAnalysis` (analysis pass)
   - Declare `StateIndependentParallelize` (transformation pass)
   - Register both in build system

6. **Implement transformation pass driver**
   - Query the `StateAccessAnalysis` for each loop
   - Use analysis result to determine which loops can be parallelized
   - Apply rewriting: convert `affine.for` → `affine.parallel` where safe
   - No duplicate analysis work—just use cached results
   - Optional: add attributes documenting analysis results (for debugging)

7. **Implement loop rewriting**
   - For parallelizable loops:
     - Create `affine.parallel` op with same bounds and step
     - Preserve loop body (including control flow) as-is
     - Update SSA operands if needed (unlikely for parallelizable loops)
   - Erase original `affine.for`
   - Run MLIR verifier on transformed function

### Part C: Build & Test

8. **Build system integration**
   - New `src/transform/StateIndependentParallelization/CMakeLists.txt`
   - Register both passes (analysis + transformation)
   - Register new library in parent `src/transform/CMakeLists.txt`
   - Link into `spmcc` executable via `src/CMakeLists.txt`

9. **Test suite**
   - Create `test/mlir/state-independent-parallelize.mlir` with:
     - **Parallelizable cases**:
       - Loop over external data with disjoint state keys (e.g., `state[pkt.srcip]`)
       - Loop with control flow conditionals (filters)
       - Loop with independent memref regions across iterations
     - **Non-parallelizable cases** (should NOT transform):
       - Loops with loop-carried dependencies (e.g., accumulation)
       - Loops with shared state access at same address
       - Loops with control flow merging state from different paths
     - **Should be ignored**:
       - Non-affine loops
       - Loops with unanalyzable index expressions

## Verification

- Run analysis only: `spmcc -pass-pipeline="builtin.module(loop-state-access-analysis)" < test/mlir/state-independent-parallelize.mlir`
- Run full pipeline: `spmcc -pass-pipeline="builtin.module(loop-state-access-analysis, state-independent-parallelize)" < test/mlir/state-independent-parallelize.mlir`
- Verify `affine.for` → `affine.parallel` transformations occur for parallelizable loops
- Verify non-parallelizable loops remain unchanged
- Verify loop-carried dependencies and shared state access prevent transformation
- Run MLIR verifier to ensure generated IR is valid
- Add `-debug-only=loop-state-access-analysis` to see analysis details: which loops are parallelizable, why others aren't
- Correctness: `affine.parallel` should execute iterations in any order (or concurrently) with identical results

## Decisions

- **Architecture**: Separate Analysis pass (reusable, cached) from Transformation pass (simple, query-based). Follows standard MLIR patterns.
- **Analysis Scope**: Module-level analysis computing results for all loops; each transformation pass queries as needed.
- **Generalization**: State independence via disjoint memory regions, not pattern matching. Handles SCR as natural special case.
- **Memory Model**: Standard MLIR memref + SPMC queue operations; analysis works on access patterns.
- **Conservatism**: When analysis inconclusive → don't parallelize (safety first).
- **Output Format**: `affine.parallel` (better compiler integration than SCF alternatives).
- **Testability**: Analysis results can be dumped/tested independently from transformation.

## Open Questions

- Should analysis provide detailed diagnostics (e.g., "Loop at line X not parallel: memref A and B aliased at index f(i)")?
- Should we add optional attributes to memrefs hinting at access patterns (non-aliasing regions) for faster analysis?
- How to handle nested loops? Parallelize all; parallelize only innermost; or decide per-loop?
- For SCR: should we add heuristic fast-path for common `(index + j) % NUM_META` pattern?
