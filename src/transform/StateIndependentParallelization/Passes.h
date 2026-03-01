#ifndef SRC_TRANSFROM_STATEINDEPENDENTPARALLELIZATION_PASSES_H_
#define SRC_TRANSFROM_STATEINDEPENDENTPARALLELIZATION_PASSES_H_

namespace mlir {
namespace spmc {

#define GEN_PASS_REGISTRATION
#include "src/transform/StateIndependentParallelization/Passes.h.inc"

} // namespace spmc
} // namespace mlir

#endif // SRC_TRANSFROM_STATEINDEPENDENTPARALLELIZATION_PASSES_H_