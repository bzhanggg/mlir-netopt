#ifndef TRANSFORM_PASSES_H_
#define TRANSFORM_PASSES_H_

#include "DeadQueueElimination.h"

namespace mlir {
namespace spmc {

#define GEN_PASS_REGISTRATION
#include "src/transform/DeadQueueElimination/Passes.h.inc"

} // namespace spmc
} // namespace mlir

#endif // TRANSFORM_PASSES_H_