#include "src/dialect/SpmcOps.h"
#include "src/dialect/SpmcTypes.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <string>

namespace mlir {
namespace spmc {

llvm::LogicalResult CreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  CreateOp::Adaptor adaptor(operands, attributes, properties, regions);
  Type eltType = adaptor.getElement();
  unsigned capacity = adaptor.getCapacity();

  QueueType qType = QueueType::get(context, eltType, capacity);
  inferredReturnTypes.emplace_back(qType);
  return success();
}

// TODO: remove this. This is bad because size should not be checked only when
// calling the size op. Instead, find a way to store the size value as an
// attribute of QueueType.
llvm::LogicalResult SizeOp::verify() {
  QueueType qType = llvm::dyn_cast<QueueType>(getQueue().getType());
  if (!qType)
    return emitOpError("expected queue type for operand");
  unsigned capacity = qType.getCapacity();

  Value size = getSize();
  if (auto constOp = size.getDefiningOp<arith::ConstantIntOp>()) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue())) {
      uint32_t sizeVal = intAttr.getUInt();
      if (sizeVal > capacity) {
        return emitOpError("queue size " + std::to_string(sizeVal) + "cannot be larger than capacity " + std::to_string(capacity));
      }
    }
  }
  return success();
}

llvm::LogicalResult PopOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  PopOp::Adaptor adaptor(operands, attributes, properties, regions);
  QueueType qType =
      llvm::dyn_cast<spmc::QueueType>(adaptor.getQueue().getType());

  if (!qType)
    return emitOptionalError(location, "Expected queue operand");

  inferredReturnTypes.emplace_back(qType.getElement());
  return success();
}

} // namespace spmc
} // namespace mlir