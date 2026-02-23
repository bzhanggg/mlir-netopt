#include "src/dialect/SpmcOps.h"
#include "src/dialect/SpmcTypes.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

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

llvm::LogicalResult CreateOp::verify() {
  if (!getElement()) {
    return emitOpError("Missing element attribute");
  }
  return success();
}

llvm::LogicalResult PushOp::verify() {
  QueueType qType = llvm::dyn_cast<QueueType>(getQueue().getType());
  if (!qType) {
    return emitOpError("expected queue type for first operand");
  }
  Type elementType = qType.getElement();
  Type valueType = getValue().getType();

  if (valueType != elementType) {
    std::string valueTypeStr;
    std::string elementTypeStr;
    llvm::raw_string_ostream valueOs(valueTypeStr);
    llvm::raw_string_ostream elementOs(elementTypeStr);
    valueType.print(valueOs);
    elementType.print(elementOs);
    return emitOpError() << "value type " << valueTypeStr
                         << " does not match queue element type "
                         << elementTypeStr;
  }
  return success();
}

llvm::LogicalResult
PopOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  PopOp::Adaptor adaptor(operands, attributes, properties, regions);
  QueueType qType =
      llvm::dyn_cast<spmc::QueueType>(adaptor.getQueue().getType());

  if (!qType)
    return emitOptionalError(location, "Expected queue operand");
  // Pop is non-blocking and returns a success flag followed by the value.
  // Emit the i1 success type first, then the element type from the queue.
  inferredReturnTypes.emplace_back(IntegerType::get(context, 1));
  inferredReturnTypes.emplace_back(qType.getElement());
  return success();
}

llvm::LogicalResult PopOp::verify() {
  QueueType qType = llvm::dyn_cast<spmc::QueueType>(getQueue().getType());
  if (!qType) {
    return emitOpError("expected queue type for operand");
  }

  Type elementType = qType.getElement();
  Type valueType = getValue().getType();

  if (elementType != valueType) {
    std::string valueTypeStr;
    std::string elementTypeStr;
    llvm::raw_string_ostream valueOs(valueTypeStr);
    llvm::raw_string_ostream elementOs(elementTypeStr);
    valueType.print(valueOs);
    elementType.print(elementOs);
    return emitOpError() << "value type " << valueTypeStr
                         << " does not match queue element type "
                         << elementTypeStr;
  }

  return success();
}

} // namespace spmc
} // namespace mlir