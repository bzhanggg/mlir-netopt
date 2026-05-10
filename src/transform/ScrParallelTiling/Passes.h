#ifndef SRC_TRANSFORM_SCRPARALLELTILING_PASSES_H_
#define SRC_TRANSFORM_SCRPARALLELTILING_PASSES_H_

#include <mlir/Pass/Pass.h>
#include "ScrParallelTiling.h"

namespace mlir {
namespace spmc {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "src/transform/ScrParallelTiling/Passes.h.inc"

} // namespace spmc
} // namespace mlir

#endif // SRC_TRANSFORM_SCRPARALLELTILING_PASSES_H_