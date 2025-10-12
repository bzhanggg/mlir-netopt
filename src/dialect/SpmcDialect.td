#ifndef SPMC_DIALECT_TD_
#define SPMC_DIALECT_TD_

include "mlir/IR/OpBase.td"

def Spmc_Dialect : Dialect {
    let name = "spmc";
    let summary = "A single producer, multiple consumer queue with thread-safe push/pop operations";
    let description = [{
        The Spmc dialect exposes APIs for safely pushing/popping from a ring buffer queue, with support
        for multiple consumers and a single producer. It allows you to create queues of fixed size, standard
        thread-safe push/pop operations, and a non-thread-safe print operation.
    }];

    let cppNamespace = "::mlir::spmc";
    let useDefaultTypePrinterParser = 1;
}

#endif // SPMC_DIALECT_TD_