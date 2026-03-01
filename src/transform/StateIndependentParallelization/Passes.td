#ifndef SRC_TRANSFROM_STATEINDEPENDENTPARALLELIZATION_PASSES_TD_
#define SRC_TRANSFROM_STATEINDEPENDENTPARALLELIZATION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def StateIndependentParallelization : Pass<"state-ind-parallel"> {
    let summary = "Automatically parallelize operations that do not access shared state";
    let description = [{
        Performs autoparallelization of operations that do not depend on shared state.
    }];
    let dependentDialects = ["mlir::spmc::SpmcDialect"];
}

#endif // SRC_TRANSFROM_STATEINDEPENDENTPARALLELIZATION_PASSES_TD_