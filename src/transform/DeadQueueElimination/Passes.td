#ifndef SRC_TRANSFORM_DEADQUEUEELMINATION_PASSES_TD_
#define SRC_TRANSFORM_DEADQUEUEELMINATION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DeadQueueElimination : Pass<"dead-queue-elimination"> {
    let summary = "Eliminate unused queues";
    let description = [{
        Perform dead code elimination on spmc.create ops
    }];
    let dependentDialects = ["mlir::spmc::SpmcDialect"];
}

#endif // SRC_TRANSFORM_DEADQUEUEELMINATION_PASSES_TD_