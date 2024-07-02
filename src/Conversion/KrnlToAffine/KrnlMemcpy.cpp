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
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
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

  static const int64_t undefMod = -1;

  // hi alex: make sure we only replace when this is true.
  static bool matchReplacementPattern(Value dest, Value src, Value len) {
    // Only accept memref of scalar types
    MemRefType destType = mlir::cast<MemRefType>(dest.getType());
    MemRefType srcType = mlir::cast<MemRefType>(src.getType());
    // Handle only scalar.
    if (MathBuilder::isVector(srcType) || MathBuilder::isVector(destType))
      return false;

    // Both lowest dim map has to be friendly.
    int64_t srcMod, destMod;
    if (!isMappedLowestDimIdentityOrModConst(srcType, srcMod))
      return false;
    if (!isMappedLowestDimIdentityOrModConst(destType, destMod))
      return false;
    // If either src or dest mod is undefined, then we don't care about the
    // actual value of the other mod. If both are defined, so far we require
    // both to be the same value. This could be relaxed to one being a multiple
    // of the other. Ignore this case for now.
    if (srcMod != undefMod && destMod != undefMod && srcMod != destMod)
      return false;

    // if we have a defined mod and literal len, make sure len<=mod.
    int64_t mod = srcMod != undefMod ? srcMod : destMod;
    IndexExpr lenIE = SymIE(len);

    if (mod != undefMod && lenIE.isLiteral()) {
      assert(lenIE.getLiteral() <= mod && "memcpy cannot span mapped memory");
    }

    // All good.
    return true;
  }

#if 0
  Value flattenMem(MDBuild &create, Value memref, DimsExpr dims, int64_t mod,
      IndexExpr offset, IndexExpr &flatOffset1, IndexExpr &flatOffset2) const {
    if (mod == undefMod) {
      flatOffset1 = offset;
      flatOffset2 = LitIE(0); // Should not be used.
      DimsExpr flatDims;
      return create.mem.reshapeToFlatInnermost(
          memref, dims, flatDims, dims.size());
    }
    // Have a friendly mapping
    srcCopyOffset1 = srcCopyOffset.ceilDiv(srcMod);
    srcCopyOffset2 = srcCopyOffset % srcMod;
    IndexExpr T = LitIE(2);
    IndexExpr M = LitIE(srcMod);
    DimsExpr reallocTileDims = {T, M};
    srcCopyValue =
        create.mem.reinterpretCast(src, litZero.getValue(), reallocTileDims);

  }
#endif

  // Only for element type source/dest.
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MDBuild create(rewriter, loc);
    IndexExprScope outerScope(create.affineKMem);

    // Get op info.
    auto krnlOp = llvm::cast<KrnlMemcpyOp>(op);
    KrnlMemcpyOpAdaptor operandAdaptor(krnlOp);
    Value src = operandAdaptor.getSrc();
    Value dest = operandAdaptor.getDest();
    Value lenInt = operandAdaptor.getNumElems(); // int64;
    assert(matchReplacementPattern(dest, src, lenInt) && "expected match");
    IndexExpr len = SymIE(operandAdaptor.getNumElems());              // int64
    IndexExpr srcInitOffset = SymIE(operandAdaptor.getSrcOffset());   // index
    IndexExpr destInitOffset = SymIE(operandAdaptor.getDestOffset()); // index
    IndexExpr zero = LitIE(0);

    // Type and vector type.
    // hi alex MemRefType srcType = mlir::cast<MemRefType>(src.getType());
    MemRefType destType = mlir::cast<MemRefType>(dest.getType());
    Type elementType = destType.getElementType();
    int64_t VL = create.vec.getMachineVectorLength(elementType);
    VectorType vecType = VectorType::get({VL}, elementType);
    int64_t U = 2;

// Mapping.
#if 0
    int64_t srcMod, destMod;
    isMappedLowestDimIdentityOrModConst(srcType, srcMod);
    isMappedLowestDimIdentityOrModConst(destType, destMod);
    int64_t mod = srcMod != undefMod ? srcMod : destMod;
#endif

    // Mem dimensions.
    IndexExpr srcMinSize = srcInitOffset + len;
    DimsExpr srcFlatDims = {srcMinSize};
    Value srcFlat =
        create.mem.reinterpretCast(src, zero.getValue(), srcFlatDims);

    // Same for dest.
    IndexExpr destMinSize = destInitOffset + len;
    DimsExpr destFlatDims = {destMinSize};
    Value destFlat =
        create.mem.reinterpretCast(dest, zero.getValue(), destFlatDims);

    int B = U * VL;
    IndexExpr lb = zero;
    IndexExpr ub = len - LitIE(B - 1);
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
    IndexExpr remainingElements = len % B;
    ub = len;
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
