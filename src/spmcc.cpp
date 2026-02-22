#include "dialect/SpmcDialect.h"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

namespace cl = llvm::cl;

namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
  "emit", cl::desc("Select the kind of output desired"),
  cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
  cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::spmc::SpmcDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "spmc compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    llvm::outs() << "Dumping AST here...\n";
    break;
  case Action::DumpMLIR:
    llvm::outs() << "Dumping MLIR here...\n";
    break;
  default:
    llvm::errs() << "No action specified, usage: -emit=<action>\n";
  }

  return 0;
}