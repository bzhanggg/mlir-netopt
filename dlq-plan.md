# Plan: Implement DeadQueueElimination Pass

Eliminate `spmc.create` operations that are never used by push/pop operations, reducing memory overhead in network packet processing.

## Steps

1. Create pass infrastructure file ([src/dialect/passes/DeadQueueElimination.cpp](src/dialect/passes/DeadQueueElimination.cpp) and header)
2. Define the `DeadQueueEliminationPass` class inheriting from MLIR's `OperationPass<func::FuncOp>`
3. Implement use-def analysis to track all queue operands and results
4. Build algorithm: walk all operations and mark queues used by push_back/pop_front
5. Eliminate unmarked queues with proper side-effect handling
6. Add pass registration to [CMakeLists.txt](CMakeLists.txt) and registration function
7. Create pass registration header for dialect integration
8. Test with expanded test suite including edge cases

## Algorithm Details

**Data structures:**

- `SmallPtrSet<Value>` — tracks live (used) queue values
- `SmallVector<Operation*>` — collects dead create operations to erase

**Walkthrough:**

1. Walk all operations in function body
2. For each `spmc.push_back` or `spmc.pop_front`, mark operand queue as live
3. For each `spmc.create`, check if its result is in live set
4. Collect all dead creates
5. Erase dead operations (handle potential dominance issues)

**Edge cases to handle:**

- Block arguments (function parameters) — never eliminate
- Operations in different blocks — use SSA dominator analysis if needed
- Potential side effects (already marked via traits)
- Queue used as function result — mark as live via return ops
- Dead queue results used elsewhere — don't eliminate if result has uses

## Further Considerations

1. **Data flow sensitivity** — Should pass consider inter-procedural analysis? Current plan is function-scoped only (conservative, safe)
2. **Performance** — Linear walk is sufficient for typical sizes; consider if need to iterate to fixed point
3. **Debug info** — Preserve location information for eliminated operations in diagnostics
