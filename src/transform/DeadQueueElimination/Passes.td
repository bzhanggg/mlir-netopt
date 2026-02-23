#ifndef SRC_TRANSFORM_DEADQUEUEELMINATION_PASSES_TD_
#define SRC_TRANSFORM_DEADQUEUEELMINATION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DeadQueueElimination : Pass<"dead-queue-elimination"> {
    let summary = "Eliminate dead SPMC queues";
    let description = [{
        Performs dead code elimination on spmc.create operations. Eliminates:
        1. Completely unused queues (zero operations on the queue result)
        2. Write-only queues (created and pushed to, but never popped from and don't escape)
        
        This optimization reduces memory overhead in SPMC-based packet processing by removing 
        queues that serve no functional purpose.
    }];
    let dependentDialects = ["mlir::spmc::SpmcDialect"];
}

#endif // SRC_TRANSFORM_DEADQUEUEELMINATION_PASSES_TD_