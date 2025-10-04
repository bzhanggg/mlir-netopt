#ifndef SPMC_OPS_TD_
#define SPMC_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Spmc_Dialect : Dialect {
    let name = "spmc";
    let summary = "A single producer, multiple consumer queue with thread-safe push/pop operations";
    let description = [{
        The Spmc dialect exposes APIs for safely pushing/popping from a ring buffer queue, with support
        for multiple consumers and a single producer. It allows you to create queues of fixed size, standard
        thread-safe push/pop operations, and a non-thread-safe print operation.
    }];

    let cppNamespace = "::mlir::spmc";
}

// Inherits from base TypeDef class. Provides:
//  * Parent dialect of the type
//  * Fully qualified name of the type
//  * Mnemonic for the type (simplified name for internal reference)
//  * A list of traits for the type
class Spmc_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Spmc_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QueueType : Spmc_Type<"Queue", "queue"> {
    let summary = "Queue type with fixed capacity and element type";
    let description = [{
        Queue types have a designated 32-bit capacity, and an element type determined
        at compile-time.
    }];

    let parameters = (ins
        TypeAttr:$type,             // i64, f32, etc.
        UI32Attr:$capacity
    );

    // Example output: !spmc.queue<i32,16>
    let assemblyFormat = "`<` $type `,` $capacity `>`";
    let genVerifyDecl = 1;
    let genMnemonicAlias = 1;
}

// Inherits from base Op class in OpBase.td. Provides:
//  * Parent dialect of the operation
//  * Mnemonic for the operation (name w/o dialect prefix)
//  * A list of traits for the operation
class Spmc_Op<string mnemonic, list<Trait> traits = []> : Op<Spmc_Dialect, mnemonic, traits>;

def CreateOp : Spmc_Op<"create", [Pure]> {
    let summary = "create queue";
    let description = [{
        Create operations instantiates a queue (ring buffer) with a fixed capacity and element type,
        both attached to the operation as attributes. For example:

        ```mlir
            %q = spmc.create : !spmc.queue<element = i32, capacity = 16>;
        ```
    }];

    let arguments = (ins
        TypeAttr:$type,
        UI32Attr:$capacity
    );

    let results = (outs QueueType);
    let hasVerifier = 1;
}

#endif // SPMC_OPS_TD_