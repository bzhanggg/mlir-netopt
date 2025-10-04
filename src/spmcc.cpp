#include "spmc/SpmcDialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include <llvm/Support/raw_ostream.h>

namespace cl = llvm::cl;

namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    llvm::cl::ParseCommandLineOptions(argc, argv, "spmc compiler\n");

    switch (emitAction) {
    case Action::DumpAST:
        llvm::outs() << "Dumping AST here...";
    case Action::DumpMLIR:
        llvm::outs() << "Dumping MLIR here...";
    default:
        llvm::errs() << "No action specified, usage: -emit=<action>\n";
    }

    return 0;
}