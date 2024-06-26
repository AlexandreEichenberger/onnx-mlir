/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- KrnlGetLinearOffsetIndex.cpp - -----------------------===//
//
// Copyright 2024- The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlMemcpyOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlMemcpyOpLoweringToAffine : public ConversionPattern {

public:
  explicit KrnlMemcpyOpLoweringToAffine(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlMemcpyOp::getOperationName(), 1, context) {}

  using MDBuild = MultiDialectBuilder<IndexExprBuilderForKrnl,
      AffineBuilderKrnlMem, VectorBuilder, MemRefBuilder, MathBuilder>;

#if 0
  static bool matchReplacementPattern(
      KrnlMemcpyOp memcpyOp, Value dest, Value len) {
    // Only accept memref of scalar types
    MemRefType type = mlir::cast<MemRefType>(dest.getType());
    // Handle only scalar.
    if (MathBuilder::isVector(type))
      return false;
    // Currently only generate explicit loop copy for arrays with literal
    // length.
    IndexExpr lenIE = SymIE(len);
    if (!lenIE.isLiteral())
      return false;
    return true;
  }
#endif

  // Only for element type source/dest.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MDBuild create(rewriter, loc);
    IndexExprScope outerScope(create.affineKMem);

    // Get info
    auto krnlOp = llvm::cast<KrnlMemcpyOp>(op);
    KrnlMemcpyOpAdaptor operandAdaptor(krnlOp);
    Value src = operandAdaptor.getSrc();
    Value dest = operandAdaptor.getDest();
    Value lenInt = operandAdaptor.getNumElems(); // int64;
    // hi alex, don't think its needed
    // Value lenIndex = create.math.castToIndex(lenInt);
    IndexExpr lenIE = SymIE(lenInt);

    // Type and vector type.
    MemRefType type = mlir::cast<MemRefType>(dest.getType());
    Type elementType = type.getElementType();
    int64_t VL = create.vec.getMachineVectorLength(elementType);
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t U = 2;

    // Flatten src & dest memrefs to 1D.
    DimsExpr srcDims, destDims, srcFlatDims, destFlatDims;
    create.krnlIE.getShapeAsSymbols(src, srcDims);
    create.krnlIE.getShapeAsSymbols(dest, destDims);
    Value srcFlat = create.mem.reshapeToFlatInnermost(
        src, srcDims, srcFlatDims, srcDims.size());
    Value destFlat = create.mem.reshapeToFlatInnermost(
        dest, destDims, destFlatDims, destDims.size());

    // Offsets from memcpy.
    IndexExpr srcInitOffset = SymIE(operandAdaptor.getSrcOffset());
    IndexExpr destInitOffset = SymIE(operandAdaptor.getDestOffset());

    int B = U * VL;
    IndexExpr lb = LitIE(0);
    IndexExpr ub = lenIE - LitIE(B - 1);
    create.affineKMem.forIE(lb, ub, B, [&](AffineBuilderKrnlMem &c, Value i) {
      MDBuild create(c);
      IndexExprScope innerScope(create.mem, &outerScope);
      IndexExpr ii = DimIE(i);
      IndexExpr srcOffset = SymIE(srcInitOffset) + ii;
      IndexExpr destOffset = SymIE(destInitOffset) + ii;
      // Guaranteed full iterations, manually unroll for more ILP.
      Value tmp[U];
      Value litOffset[U];
      // Load U * VL simd values.
      for (int u = 0; u < U; ++u) {
        litOffset[u] = create.math.constantIndex(u * VL);
        tmp[u] =
            create.vec.loadIE(vecType, srcFlat, {srcOffset}, {litOffset[u]});
      }
      // Store U * VL simd values.
      for (int u = 0; u < U; ++u) {
        create.vec.storeIE(tmp[u], destFlat, {destOffset}, {litOffset[u]});
      }
    });
    // Elements not covered by blocked loop.
    IndexExpr remainingElements = lenIE % B;
    ub = lenIE;
    lb = ub - remainingElements;
    create.affineKMem.forIE(lb, ub, 1, [&](AffineBuilderKrnlMem &c, Value i) {
      MDBuild create(c);
      IndexExprScope innermostScope(create.mem, &outerScope);
      IndexExpr ii = DimIE(i);
      IndexExpr srcOffset = SymIE(srcInitOffset) + ii;
      IndexExpr destOffset = SymIE(destInitOffset) + ii;
      Value tmp = create.affineKMem.loadIE(srcFlat, {srcOffset}, {});
      create.affineKMem.storeIE(tmp, destFlat, {destOffset}, {});
    });
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlMemcpyOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlMemcpyOpLoweringToAffine>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
