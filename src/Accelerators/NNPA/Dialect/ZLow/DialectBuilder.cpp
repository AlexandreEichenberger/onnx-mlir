/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-------------- DialectBuilder.cpp - Krnl Dialect Builder ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ZLow Builder
//===----------------------------------------------------------------------===//

#if 0
// Supports vectors that have multiple of 8 dlf16 values.
Value ZLowBuilder::convertDLF16ToF32Vector(mlir::Value dlf16Vals) const {
  int64_t archVL = 8;              // FP16 archVL.
  int64_t archVLHalf = archVL / 2; // FP32 archVL.
  Type f16Type = b().getF16Type();
  Type f32Type = b().getF32Type();

  MultiDialectBuilder<MathBuilder, VectorBuilder> create(*this);

  VectorType vecF16Type = mlir::cast<VectorType>(dlf16Vals.getType());
  int64_t outerDim;
  Value df16Vals2D = create.vec.shapeCast2D(dlf16Vals, outerDim, archVL);
}

// Supports vectors that have multiple of 8 f32 values.
Value ZLowBuilder::convertF32ToDLF16Vector(mlir::Value f32Vals) const {}

#endif

// =============================================================================
// IndexExpr Builder for Analysis
// =============================================================================

// Return null if none is found.
ElementsAttr IndexExprBuilderForZLow::getConst(Value value) { return nullptr; }

Value IndexExprBuilderForZLow::getVal(Value intArrayVal, uint64_t i) {
  MultiDialectBuilder<AffineBuilder, MathBuilder> create(*this);
  uint64_t rank = getShapedTypeRank(intArrayVal);
  if (rank == 0)
    return create.affine.load(intArrayVal);
  uint64_t size = getArraySize(intArrayVal);
  assert(i < size && "out of bound reference");
  Value iVal = create.math.constantIndex(i);
  return create.affine.load(intArrayVal, {iVal});
}

Value IndexExprBuilderForZLow::getShapeVal(
    Value tensorOrMemrefValue, uint64_t i) {
  MemRefBuilder createMemRef(*this);
  return createMemRef.dim(tensorOrMemrefValue, i);
}

} // namespace onnx_mlir
