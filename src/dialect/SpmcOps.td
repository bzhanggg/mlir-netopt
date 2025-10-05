#ifndef SPMC_OPS_TD_
#define SPMC_OPS_TD_

include "SpmcDialect.td"
include "SpmcTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Inherits from base Op class in OpBase.td. Provides:
//  * Parent dialect of the operation
//  * Mnemonic for the operation (name w/o dialect prefix)
//  * A list of traits for the operation
class Spmc_Op<string mnemonic, list<Trait> traits = []> : Op<Spmc_Dialect, mnemonic, traits>;

def CreateOp : Spmc_Op<"create", [Pure, InferTypeOpInterface]> {
    let summary = "create queue";
    let description = [{
        Create operation instantiates a queue (ring buffer) with a fixed capacity and element type,
        both attached to the operation as attributes. For example:

        ```mlir
            %q = spmc.create : !spmc.queue<element = i32, capacity = 16>;
        ```
    }];

    let arguments = (ins
        TypeAttr:$element,
        UI32Attr:$capacity
    );

    let results = (outs QueueType);
    let hasVerifier = 1;
}

def PushOp : Spmc_Op<"push_back"> {
    let summary = "push";
    let description = [{
        Push_back operation pushes a value of type T onto the back of a queue with element type T
        in a thread-safe manner. This is used by the single producer to add new values to the queue,
        when the queue size is less than its capacity. For example:

        ```mlir
            spmc.push_back %q, %val : !spmc.queue<i32, 16>
        ```
    }];

    let arguments = (ins
        QueueType:$queue,
        AnyType:$value
    );

    let assemblyFormat = "$queue `,` $value attr-dict `:` `(` type($queue) `,` type($value) `)`";

    // verifies that $value shares the same type as $queue .element
    let hasVerifier = 1;
}

def PopOp : Spmc_Op<"pop_front", [InferTypeOpInterface]> {
    let summary = "pop";
    let description = [{
        Pop_front operation pops a value of type T from the front of a queue with element type T
        in a thread-safe manner. This can be called by multiple consumers to remove value from the queue,
        when the queue size is greater than zero. For example:

        ```
            spmc.pop_front %q : !spmc.queue<i32, 16>
        ```
    }];

    let arguments = (ins QueueType:$queue);
    let results = (outs AnyType:$value);

    let assemblyFormat = "$queue attr-dict `:` type($queue) `->` qualified(type($value))";

    // verifies that $value shares the same type as $queue .element
    let hasVerifier = 1;
}

#endif // SPMC_OPS_TD_