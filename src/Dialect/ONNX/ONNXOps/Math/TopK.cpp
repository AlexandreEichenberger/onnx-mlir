/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ TopK.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect TopK operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <>
LogicalResult ONNXTopKOpShapeHelper::computeShape() {
  DimsExpr outputDims;
  ONNXTopKOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
  // Get info about X and K operands.
  Value X = operandAdaptor.X();
  Value K = operandAdaptor.K();
  int64_t rank = createIE->getShapedTypeRank(X);

  // Axis to compute TopK.
  int64_t axis = operandAdaptor.axis();
  axis = axis < 0 ? axis + rank : axis;
  assert(axis >= 0 && axis < rank && "axis is out of bound");

  // K is a scalar tensor storing the number of returned values along the given
  // axis.
  IndexExpr kIE = createIE->getIntAsSymbol(K);
  if (kIE.isUndefined())
    return op->emitError("K input parameter could not be processed");

  // If K is literal, it must be less than the axis dimension size.
  IndexExpr XAxisDim = createIE->getShapeAsDim(X, axis);
  if (kIE.isLiteral() && XAxisDim.isLiteral())
    if (kIE.getLiteral() >= XAxisDim.getLiteral())
      return op->emitError("K value is out of bound");

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis)
      outputDims.emplace_back(kIE);
    else
      outputDims.emplace_back(createIE->getShapeAsDim(X, i));
  }

  setOutputDims(outputDims, 0);
  setOutputDims(outputDims, 1);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

LogicalResult ONNXTopKOp::verify() {
  ONNXTopKOpAdaptor operandAdaptor(*this);

  Value K = operandAdaptor.K();
  if (hasShapeAndRank(K)) {
    // K's rank must be zero or one.
    int64_t KRank = K.getType().cast<ShapedType>().getRank();
    if (KRank > 1)
      return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
          *this->getOperation(), K, KRank, "< 2");
  }

  // axis attribute must be in the range [-r,r-1], where r = rank(X).
  Value X = operandAdaptor.X();
  if (hasShapeAndRank(X)) {
    int64_t Xrank = X.getType().cast<ShapedType>().getRank();
    int64_t axis = this->axis();

    if (axis < -Xrank || axis >= Xrank)
      return onnx_mlir::Diagnostic::emitAttributeOutOfRangeError(
          *this->getOperation(), "axis", axis,
          onnx_mlir::Diagnostic::Range<int64_t>(-Xrank, Xrank - 1));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper *ONNXTopKOp::getShapeHelper(Operation *op,
    ArrayRef<mlir::Value> oper, IndexExprBuilder *ieb, IndexExprScope *scope) {
  return getNewShapeHelper<ONNXTopKOpShapeHelper>(op, oper, ieb, scope);
}

LogicalResult ONNXTopKOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer the output shape if the operands shape isn't known yet.
  if (llvm::any_of(this->getOperands(),
          [](const Value &op) { return !hasShapeAndRank(op); }))
    return success();

  Builder b(getContext());
  Type elementType = X().getType().cast<ShapedType>().getElementType();
  ONNXTopKOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateTypes({elementType, b.getI64Type()});
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXTopKOp>;
} // namespace onnx_mlir
