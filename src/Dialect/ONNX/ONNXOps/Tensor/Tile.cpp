/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Tile.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Tile operation.
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
LogicalResult ONNXTileOpShapeHelper::computeShape() {
  ONNXTileOpAdaptor operandAdaptor(operands);
  // Get info about input data operand.
  Value input = operandAdaptor.getInput();
  int64_t inputRank = createIE->getShapedTypeRank(input);
  Value repeats = operandAdaptor.getRepeats();
  fprintf(stderr, "hi alex: input and repeat\n");
  input.dump();
  repeats.dump();
  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  fprintf(stderr, "hi alex, input rank is %d\n", (int) inputRank);
  for (int64_t i = 0; i < inputRank; i++) {
    fprintf(stderr, "\nhi alex, start iter %d\n", (int) i);
    IndexExpr dimInput = createIE->getShapeAsDim(input, i);
    dimInput.debugPrint("  dim input from tile");
    IndexExpr repeatsValue =
        createIE->getIntFromArrayAsSymbol(repeats, i, inputRank);
    fprintf(stderr, "  done computing repeat value\n");
    repeatsValue.debugPrint("  repeat value");
    outputDims[i] = dimInput * repeatsValue;
    fprintf(stderr, "hi alex, done with iter %d\n\n", (int) i);
  }
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXTileOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getInput()) || !hasShapeAndRank(getRepeats()))
    return success();

  // 'repeats' tensor is an 1D tensor.
  auto repeatsTensorTy = getRepeats().getType().cast<RankedTensorType>();
  if (repeatsTensorTy.getShape().size() != 1)
    return emitError("Repeats tensor must have rank one");

  Type elementType = getInput().getType().cast<ShapedType>().getElementType();
  ONNXTileOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
template struct ONNXNonSpecificOpShapeHelper<ONNXTileOp>;
} // namespace onnx_mlir
