/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- FusedOpLoweringBase.hpp - Generic FusedOp lowering base -----===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Non-accelerator infrastructure for lowering ONNXFusedOp.
//
// FusedOpKindLowering<FusionT>
//   Template base for per-kind lowering patterns.  Subclasses only implement
//   lowerVerified(), returning the Value that replaces the FusedOp; the base
//   handles kind dispatch, retrieve/verify, the inline fallback, and (on
//   success) dropping the body's intra-region def-use edges and replacing the
//   op.  Register one subclass per FusedOp kind.
//
// FusedOpInlineFallback
//   Benefit-0 catch-all registered in the general ONNX→Krnl pass.  Fires for
//   any ONNXFusedOp whose kind has no dedicated per-kind pattern registered in
//   the current pass.  Inlines the body so constituent ops can be lowered
//   individually.  Independent of any accelerator.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_FUSED_OP_LOWERING_BASE_H
#define ONNX_MLIR_FUSED_OP_LOWERING_BASE_H

#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/FusionOpChain.hpp"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// FusedOpKindLowering<FusionT>
//
// Usage:
//   struct MyLowering : public FusedOpKindLowering<MyFusion> {
//     MyLowering(TypeConverter &tc, MLIRContext *ctx, ...)
//         : FusedOpKindLowering(tc, ctx) { ... }
//
//     FailureOr<Value> lowerVerified(ONNXFusedOp, OpAdaptor,
//         ConversionPatternRewriter &, MyFusion &) const override { ... }
//   };
//===----------------------------------------------------------------------===//

template <typename FusionT>
struct FusedOpKindLowering
    : public mlir::OpConversionPattern<mlir::ONNXFusedOp> {
  using Base = mlir::OpConversionPattern<mlir::ONNXFusedOp>;
  using OpAdaptor = typename mlir::ONNXFusedOp::Adaptor;

  using Base::Base; // inherit TypeConverter + MLIRContext constructors

  mlir::LogicalResult matchAndRewrite(mlir::ONNXFusedOp fusedOp,
      OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    FusionT fusion;
    // Only handle our kind; return failure() so the correct per-kind pattern
    // (or the benefit-0 catch-all) can process the op instead.
    if (fusedOp.getKind() != FusionT::kKind)
      return mlir::failure();
    // Retrieve body ops and cross-check stored params.  If the body was
    // altered by an optimisation pass, fall back to inline.
    fusion.retrieveOpsAndOutputValues(fusedOp);
    if (!fusion.verifyAndRetrieveAttrs(fusedOp))
      // Failed, use the inline fallback to allow constituent ops to be
      // lowered on their own.
      return FusionOpChain::inlineFallback(rewriter, fusedOp);

    mlir::FailureOr<mlir::Value> replacement =
        lowerVerified(fusedOp, adaptor, rewriter, fusion);
    if (mlir::failed(replacement))
      return mlir::failure();

    // Drop all intra-body def-use edges before handing the FusedOp to the
    // conversion framework's deferred eraser.  Without this, the conversion
    // framework (applyRewrites / EraseBlockRewrite::cleanup) may encounter
    // ops that still appear to have uses when erased, triggering an
    // assertion in eraseSingleOp even though the RegionKindInterface marks
    // the body as a Graph region.  Dropping all result uses here makes every
    // inner op use_empty() regardless of the order in which the eraser
    // visits them.
    for (mlir::Block &block : fusedOp.getBody())
      for (mlir::Operation &innerOp : block)
        innerOp.dropAllUses();

    rewriter.replaceOp(fusedOp, *replacement);
    return mlir::success();
  }

  /// Implement the actual lowering.  Called only when the kind matches and
  /// verifyAndRetrieveAttrs() succeeded; fusion fields are fully populated.
  /// Return the Value that replaces the FusedOp — do not call
  /// rewriter.replaceOp() or drop body uses; the base class does both.
  virtual mlir::FailureOr<mlir::Value> lowerVerified(mlir::ONNXFusedOp fusedOp,
      OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter,
      FusionT &fusion) const = 0;
};

//===----------------------------------------------------------------------===//
// FusedOpInlineFallback
//
// Register this at benefit=0 in every conversion pass that may see
// ONNXFusedOp.  It inlines the body so that constituent ops can be lowered
// by their own patterns within the same pass.
//===----------------------------------------------------------------------===//

struct FusedOpInlineFallback
    : public mlir::OpConversionPattern<mlir::ONNXFusedOp> {
  using OpAdaptor = typename mlir::ONNXFusedOp::Adaptor;

  FusedOpInlineFallback(
      mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::ONNXFusedOp>(
            typeConverter, ctx, /*benefit=*/0) {}

  mlir::LogicalResult matchAndRewrite(mlir::ONNXFusedOp fusedOp, OpAdaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    fusedOp.emitWarning()
        << "no dedicated lowering for onnx.Fused (kind='" << fusedOp.getKind()
        << "'); inlining body as fallback — add a FusedOpKindLowering subclass";
    return FusionOpChain::inlineFallback(rewriter, fusedOp);
  }
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_FUSED_OP_LOWERING_BASE_H
