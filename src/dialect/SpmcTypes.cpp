#include "src/dialect/SpmcTypes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Diagnostics.h>

namespace mlir {
namespace spmc {

llvm::LogicalResult
QueueType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  Type element, unsigned int capacity) {
  if (!element) {
    return emitError() << "queue element type is invalid", failure();
  }
  if (capacity == 0) {
    return emitError() << "queue capacity must be > 0", failure();
  }
  return success();
}

} // namespace spmc
} // namespace mlir