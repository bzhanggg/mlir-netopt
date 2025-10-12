#ifndef SPMC_TYPES_TD_
#define SPMC_TYPES_TD_

include "SpmcDialect.td"
include "mlir/IR/AttrTypeBase.td"

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
        "::mlir::Type":$element,             // i64, f32, etc.
        "unsigned":$capacity
    );

    // Example output: !spmc.queue<i32,16>
    let assemblyFormat = "`<` $element `,` $capacity `>`";
    let genVerifyDecl = 1;
    let genMnemonicAlias = 1;
}

#endif // SPMC_TYPES_TD_