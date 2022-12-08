/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ MatMul.cpp - ONNX Operations ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect MatMul (float, int QL)
// operation.
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

// Templates below are needed because ONNXMatMulOp and ONNXMatMulIntegerOp use
// operands A & B, but ONNXQLinearMatMulOp uses a & b.

template <typename OpAdaptor>
std::pair<Value, Value> matMulInputs(OpAdaptor &operandAdaptor) {
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  return std::pair(A, B);
}

template <>
std::pair<Value, Value> matMulInputs(
    ONNXQLinearMatMulOpAdaptor &operandAdaptor) {
  Value A = operandAdaptor.a();
  Value B = operandAdaptor.b();
  return std::pair(A, B);
}

template <typename OP_TYPE>
LogicalResult NewONNXGenericMatMulOpShapeHelper<OP_TYPE>::computeShape() {
  typename OP_TYPE::Adaptor operandAdaptor(operands);

  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
  Value A, B;
  std::tie(A, B) = matMulInputs(operandAdaptor);

  // Size all the arrays to padded length.
  uint64_t aRank = createIE->getTypeRank(A);
  uint64_t bRank = createIE->getTypeRank(B);
  int paddedRank = std::max(aRank, bRank);
  paddedRank = std::max(paddedRank, 2);
  aDims.resize(paddedRank);
  bDims.resize(paddedRank);
  aPadDims.resize(paddedRank, false);
  bPadDims.resize(paddedRank, false);

  // Add the dims of A. All of the aDim[0]...aDim[aRank-1] are in the
  // rightmost positions, prepended by 1s to fit the paddedRankSize. (1,1,1...
  // 1, aDim[0]...aDim[aRank-1])
  LiteralIndexExpr one(1);
  int aOffset = paddedRank - aRank;
  for (int i = 0; i < aOffset; ++i) {
    aDims[i] = one;
    aPadDims[i] = true;
  }
  for (unsigned int i = 0; i < aRank; ++i) {
    aDims[i + aOffset] = createIE->getShapeAsDim(A, i);
    aPadDims[i + aOffset] = false; // Pad false even if dim is sized 1.
  }
  // for B: two cases. If bRank = 1, we pad the rightmost position. Namely we
  // get (1...,1, bDim[0], 1). We use one padding credit for the rightmost
  // position. Otherwise, when bRank>1, we only pad the leading positions.
  // Namely we get (1,1,1...,1, bDim[0],.... bDim[bRank-1])
  int bOffset = paddedRank - bRank;
  if (bRank == 1) {
    bDims[paddedRank - 1] = one;
    bPadDims[paddedRank - 1] = true;
    bOffset--;
  }
  for (int i = 0; i < bOffset; ++i) {
    bDims[i] = one;
    bPadDims[i] = true;
  }
  for (unsigned int i = 0; i < bRank; ++i) {
    bDims[i + bOffset] = createIE->getShapeAsDim(B, i);
    bPadDims[i + bOffset] = false; // Pad false even if dim is sized 1.
  }
  assert(aDims.size() == bDims.size() && "padded A&B must have same size");

  // Fill in the output dimensions, start with the non-matmul dims.
  for (int i = 0; i < paddedRank - 2; ++i) {
    // Check for broadcast, then literals, then runtime for both.
    if (aDims[i].isLiteralAndIdenticalTo(1)) {
      // A is broadcast, use B dim.
      outputDims.emplace_back(bDims[i]);
    } else if (bDims[i].isLiteralAndIdenticalTo(1)) {
      // B is a broadcast, use A dim.
      outputDims.emplace_back(aDims[i]);
    } else if (aDims[i].isLiteral() && bDims[i].isLiteral()) {
      // No broadcast, both literals, make sure they have the same value.
      if (aDims[i].getLiteral() != bDims[i].getLiteral())
        return this->op->emitError("Incompatible size detected");
      outputDims.emplace_back(aDims[i]);
    } else if (aDims[i].isLiteral()) {
      // A dim is a literal; use it here for output and b, since b
      // is guaranteed not to be a broadcast (earlier tests).
      outputDims.emplace_back(aDims[i]);
      bDims[i] = aDims[i]; // Add runtime check if desired.
    } else if (bDims[i].isLiteral()) {
      // A dim is a literal; use it here for output and a, since a
      // is guaranteed not to be a broadcast (earlier tests).
      outputDims.emplace_back(bDims[i]);
      aDims[i] = bDims[i]; // Add runtime check if desired.
    } else {
      // Have no broadcast or literal, just pick a for output; add runtime
      // check if desired.
      outputDims.emplace_back(aDims[i]);
    }
  }
  // We now check get the last two dimensions: NxK times KxM.
  int aN = paddedRank - 2;
  int aK = paddedRank - 1;
  int bK = paddedRank - 2;
  int bM = paddedRank - 1;
  // And test the K dimensions.
  if (aDims[aK].isLiteral() && bDims[bK].isLiteral()) {
    if (aDims[aK].getLiteral() != bDims[bK].getLiteral())
      return this->op->emitError("reduction dimension must be the same");
  } else if (aDims[aK].isLiteral()) {
    // Save aK dims into bK dims, in case bK dims was runtime
    bDims[bK] = aDims[aK];
  } else if (bDims[bK].isLiteral()) {
    // Save bK dims into aK dims, in case aK dims was runtime
    aDims[aK] = bDims[bK];
  }
  // Add lower N x M dimensions if they are not padded dimensions.
  if (!aPadDims[aN])
    outputDims.emplace_back(aDims[aN]);
  if (!bPadDims[bM])
    outputDims.emplace_back(bDims[bM]);
  // For the case where both aRank == bRank == 1
  if (aRank == 1 && bRank == 1) {
    assert(outputDims.empty() && "1-D x 1-D results in scalar");
  }
  // Save the final result.
  this->setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMatMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!A().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>())
    return success();

  auto elementType = A().getType().cast<ShapedType>().getElementType();
  NewONNXMatMulOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// MatMulIntegerOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXMatMulIntegerOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!A().getType().isa<RankedTensorType>() ||
      !B().getType().isa<RankedTensorType>())
    return success();

  auto elementType = getResult().getType().cast<ShapedType>().getElementType();
  NewONNXMatMulIntegerOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// QLinearMatMulOp
//===----------------------------------------------------------------------===//

LogicalResult ONNXQLinearMatMulOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  if (!a().getType().isa<RankedTensorType>() ||
      !b().getType().isa<RankedTensorType>())
    return success();

  auto elementType = getResult().getType().cast<ShapedType>().getElementType();
  NewONNXQLinearMatMulOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct NewONNXGenericMatMulOpShapeHelper<ONNXMatMulOp>;
template struct NewONNXGenericMatMulOpShapeHelper<ONNXMatMulIntegerOp>;
template struct NewONNXGenericMatMulOpShapeHelper<ONNXQLinearMatMulOp>;

} // namespace onnx_mlir
