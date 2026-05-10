#ifndef SRC_TRANSFORM_SCRCHUNKPARALLELIZATION_PASSES_TD_
#define SRC_TRANSFORM_SCRCHUNKPARALLELIZATION_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ScrParallelTiling : Pass<"scr-parallel-tile"> {
    let summary = "Parallelize SPMC consumer loops using SCR-style tiling";
    let description = [{
        Tiles loops whose only parallelization blocker is spmc.pop_front by a
        configurable core count, then parallelizes the inner (tiled) loop.
        pop_front is safe for multiple concurrent consumers (SPMC guarantee),
        and the analysis verifies that the remaining computation is
        state-independent (no iter_args, no memory conflicts).
    }];
    let dependentDialects = ["mlir::spmc::SpmcDialect"];
    let options = [
        Option<"numCores", "num-cores", "unsigned", /*default=*/"4",
               "Number of cores to chunk across (tile size for inner loop)">
    ];
}

#endif // SRC_TRANSFORM_SCRCHUNKPARALLELIZATION_PASSES_TD_
