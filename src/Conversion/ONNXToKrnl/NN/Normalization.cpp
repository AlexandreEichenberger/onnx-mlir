/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Normalization.cpp - Lowering Normalization Ops -----------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX Normalization Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include <functional>

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Batch Norm
//===----------------------------------------------------------------------===//

struct ONNXBatchNormalizationInferenceModeOpLowering
    : public OpConversionPattern<ONNXBatchNormalizationInferenceModeOp> {
  ONNXBatchNormalizationInferenceModeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(
      ONNXBatchNormalizationInferenceModeOp batchnormOp,
      ONNXBatchNormalizationInferenceModeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // batchnorm{epsilon}(x, scale, bias, mean, variance) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    Operation *op = batchnormOp.getOperation();
    Location loc = ONNXLoc<ONNXBatchNormalizationInferenceModeOp>(op);

    MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
        rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    Value epsilon = create.math.constant(
        memRefType.getElementType(), adaptor.getEpsilon().convertToDouble());
    Value operand = adaptor.getX();
    Value scale = adaptor.getScale();
    Value bias = adaptor.getB();
    Value mean = adaptor.getMean();
    Value variance = adaptor.getVar();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = create.mem.alignedAlloc(operand, memRefType);

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn or N.
    // In case of N, C is assumed to be 1.
    // Shapes of scale, bias, mean and variance must be C.
    // Computation of BatchNormalization is done as if scale, bias, mean, and
    // variance are reshaped to Cx1x1x...x1.

    // rank
    int64_t rank = memRefType.getRank();

    std::vector<Value> originalLoops;
    defineLoops(rewriter, loc, originalLoops, rank);

    // Create a KrnlIterateOp along C dimension.
    // This will be the outer-most loop in order to re-use scale, bias,
    // mean and variance.

    SmallVector<Value, 1> loopCIVs;
    if (rank > 1) {
      // TODO use new KrnlDialectBuilder.
      krnl::KrnlIterateOperandPack cPack(rewriter, originalLoops[1]);
      addDimensionToPack(rewriter, loc, cPack, operand, 1);
      KrnlIterateOp cIterateOp = create.krnl.iterate(cPack);
      Block &cIterationBlock = cIterateOp.getBodyRegion().front();
      rewriter.setInsertionPointToStart(&cIterationBlock);
      for (auto arg : cIterationBlock.getArguments())
        loopCIVs.emplace_back(arg);
    } else
      loopCIVs.emplace_back(create.math.constantIndex(0));

    Value scaleVal = create.krnl.load(scale, loopCIVs);
    Value biasVal = create.krnl.load(bias, loopCIVs);
    Value meanVal = create.krnl.load(mean, loopCIVs);
    Value varianceVal = create.krnl.load(variance, loopCIVs);

    // Create a KrnlIterateOp along the other dimensions.
    SmallVector<int64_t, 4> axes;
    axes.emplace_back(0);
    for (int64_t i = 2; i < rank; ++i)
      axes.emplace_back(i);
    std::vector<Value> packLoops;
    for (size_t i = 0; i < axes.size(); ++i)
      packLoops.emplace_back(originalLoops[axes[i]]);

    // TODO use new KrnlDialectBuilder.
    krnl::KrnlIterateOperandPack pack(rewriter, packLoops);
    for (size_t i = 0; i < axes.size(); ++i)
      addDimensionToPack(rewriter, loc, pack, operand, axes[i]);

    KrnlIterateOp iterateOp = create.krnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&iterationBlock);

    SmallVector<Value, 4> loopIVs;
    auto args = iterationBlock.getArguments();
    if (args.size() > 1) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
      for (unsigned int i = 1; i < args.size(); ++i)
        loopIVs.emplace_back(args[i]);
    } else if (rank == 2) {
      loopIVs.emplace_back(args[0]);
      loopIVs.emplace_back(loopCIVs[0]); // Insert C back.
    } else
      loopIVs.emplace_back(args[0]);

    Value xVal = create.krnl.load(operand, loopIVs);
    // normalize
    Value dividend = create.math.sub(xVal, meanVal);
    Value adjustedVarianceVal = create.math.add(varianceVal, epsilon);
    Value divisor = create.math.sqrt(adjustedVarianceVal);
    Value normVal = create.math.div(dividend, divisor);
    // scale and shift
    Value scaleNormVal = create.math.mul(scaleVal, normVal);
    Value shiftScaleNormVal = create.math.add(scaleNormVal, biasVal);
    create.krnl.store(shiftScaleNormVal, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    onnxToKrnlSimdReport(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Instance Normalization
//===----------------------------------------------------------------------===//

struct ONNXInstanceNormalizationOpLowering
    : public OpConversionPattern<ONNXInstanceNormalizationOp> {
  ONNXInstanceNormalizationOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXInstanceNormalizationOp instanceOp,
      ONNXInstanceNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // instance_normalization{epsilon}(x, scale, bias) =
    //      scale * (x - mean) / sqrt(variance + epsilon) + bias
    Operation *op = instanceOp.getOperation();
    Location loc = ONNXLoc<ONNXInstanceNormalizationOp>(op);

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();
    Value epsilon = create.math.constant(
        elementType, adaptor.getEpsilon().convertToDouble());
    Value inputMemRef = adaptor.getInput();
    Value scaleMemRef = adaptor.getScale();
    Value biasMemRef = adaptor.getB();

    // Insert an allocation and deallocation for the result of this operation.
    Value resMemRef = create.mem.alignedAlloc(inputMemRef, memRefType);

    // Operand's dimensions can be in the form of NxCxD1xD2x...xDn
    // Shapes of scale, bias must be C.

    // Get rank, bounds, and constructors.
    int64_t rank = memRefType.getRank();
    IndexExprScope outerScope(create.krnl);
    SmallVector<IndexExpr, 4> inputBounds;
    create.krnlIE.getShapeAsSymbols(inputMemRef, inputBounds);
    MemRefType tmpType = MemRefType::get({}, elementType);
    Value fZero = create.math.constant(elementType, 0);
    Value tmpMemRef = create.mem.alloca(tmpType);

    // Compute the number of values in a single channel: product of spatial
    // dimensions, converted to float.
    IndexExpr num = inputBounds[2];
    for (int d = 3; d < rank; ++d)
      num = num * inputBounds[d];
    // Convert num to float from Pooling postProcessPoolingWindow.
    Value meanDenom = create.math.cast(elementType, num.getValue());

    // Iterate over the batch and channels.
    LiteralIndexExpr iZero(0);
    ValueRange n_c_loopDef = create.krnl.defineLoops(2);
    create.krnl.iterateIE(n_c_loopDef, n_c_loopDef, {iZero, iZero},
        {inputBounds[0], inputBounds[1]},
        [&](KrnlBuilder &ck, ValueRange n_c_loopInd) {
          MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
              ck);
          IndexExprScope channelScope(ck);
          DimIndexExpr n(n_c_loopInd[0]), c(n_c_loopInd[1]);

          // Set bounds for iterating over values in channel.
          ValueRange spatial_loopDef = create.krnl.defineLoops(rank - 2);
          SmallVector<IndexExpr, 4> lbs(rank - 2, iZero);
          SmallVector<IndexExpr, 4> ubs;
          for (int d = 2; d < rank; ++d)
            ubs.emplace_back(SymbolIndexExpr(inputBounds[d]));

          // First compute the mean: store zero in reduction value, then sum up
          // all of the values in the channel, and divide by the number of
          // values.
          create.krnl.store(fZero, tmpMemRef, {});
          // Iterate over kernel and add values.
          ValueRange spatial2_loopDef = create.krnl.defineLoops(rank - 2);
          create.krnl.iterateIE(spatial2_loopDef, spatial2_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 6> inputAccessFct = {
                    n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  inputAccessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value oldSum = create.krnl.load(tmpMemRef, {});
                Value val = create.krnl.load(inputMemRef, inputAccessFct);
                Value newSum = create.math.add(oldSum, val);
                create.krnl.store(newSum, tmpMemRef);
              });
          Value sum = create.krnl.load(tmpMemRef);
          Value mean = create.math.div(sum, meanDenom);
          // Second, compute the standard dev: sum of (val - mean)2 / (num-1).
          create.krnl.store(fZero, tmpMemRef, {});
          // Iterate over kernel and add values.
          create.krnl.iterateIE(spatial_loopDef, spatial_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 6> inputAccessFct = {
                    n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  inputAccessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value oldSum = create.krnl.load(tmpMemRef, {});
                Value val = create.krnl.load(inputMemRef, inputAccessFct);
                val = create.math.sub(val, mean);
                val = create.math.mul(val, val);
                Value newSum = create.math.add(oldSum, val);
                create.krnl.store(newSum, tmpMemRef);
              });
          sum = create.krnl.load(tmpMemRef);
          // Variance is numerically off when divided by (num -1), but
          // passes the tests when divided by num, so keep that.
          Value variance = create.math.div(sum, meanDenom);

          // Calculate ahead the scale[c] / sqrt(var + epsilon)
          Value denom = create.math.add(variance, epsilon);
          denom = create.math.sqrt(denom);
          Value nom = create.krnl.load(scaleMemRef, {c.getValue()});
          Value factor = create.math.div(nom, denom);
          Value term = create.krnl.load(biasMemRef, {c.getValue()});

          // Iterate over all channel values and compute y = factor * (x - mean)
          // + term.
          ValueRange spatial3_loopDef = create.krnl.defineLoops(rank - 2);
          create.krnl.iterateIE(spatial3_loopDef, spatial3_loopDef, lbs, ubs,
              [&](KrnlBuilder &createKrnl, ValueRange spatial_loopInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 6> accessFct = {n.getValue(), c.getValue()};
                for (int d = 0; d < rank - 2; ++d)
                  accessFct.emplace_back(spatial_loopInd[d]);
                // tmp += input[n,c, spatial dims]
                Value x = create.krnl.load(inputMemRef, accessFct);
                Value val = create.math.sub(x, mean);
                val = create.math.mul(factor, val);
                val = create.math.add(val, term);
                create.krnl.store(val, resMemRef, accessFct);
              });
        }); // For all batches, channels.

    rewriter.replaceOp(op, resMemRef);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Layer Normalization
//===----------------------------------------------------------------------===//

using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
    MemRefBuilder, MathBuilder, VectorBuilder, OnnxBuilder, AffineBuilderKrnlMem>;

static inline void replaceLayerNormalizationOp(
    ConversionPatternRewriter &rewriter, ONNXLayerNormalizationOp lnOp, Value Y,
    Value meanOfX, Value invStdDev) {
  llvm::SmallVector<Value, 3> outputs;
  outputs.emplace_back(Y);
  Value noneValue;
  if (isNoneValue(lnOp.getMean()))
    outputs.emplace_back(noneValue);
  else
    outputs.emplace_back(meanOfX);
  if (isNoneValue(lnOp.getInvStdDev()))
    outputs.emplace_back(noneValue);
  else
    outputs.emplace_back(invStdDev);
  rewriter.replaceOp(lnOp, outputs);
}

// Generate the original ONNX operations. This is the unoptimized path.
// TODO: conversions of types are not handled.
LogicalResult generateONNXLayerNormalizationOpONNXCode(
    ConversionPatternRewriter &rewriter, Location loc,
    ONNXLayerNormalizationOp lnOp) {
  MDBuilder create(rewriter, loc);
  Value X = lnOp.getX(); // Original value, not translated.
  TensorType XType = X.getType().cast<TensorType>();
  Type elementType = XType.getElementType();
  int64_t XRank = XType.getRank();
  int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
  // Get epsilon
  FloatAttr epsilonAttr = lnOp.getEpsilonAttr();
  DenseElementsAttr epsilonDenseAttr =
      onnx_mlir::createDenseElementsAttrFromFloatAttr(
          rewriter, elementType, epsilonAttr);
  Value epsilon = create.onnx.constant(epsilonDenseAttr);

  // Create reduction axes array.
  llvm::SmallVector<int64_t, 4> axesIntArray, reductionShape;
  for (int64_t r = 0; r < axis; ++r)
    reductionShape.emplace_back(XType.getShape()[r]);
  for (int64_t r = axis; r < XRank; ++r) {
    reductionShape.emplace_back(1);
    axesIntArray.emplace_back(r);
  }
  Value axes =
      create.onnx.constant(create.getBuilder().getI64TensorAttr(axesIntArray));
  TensorType reductionType = RankedTensorType::get(reductionShape, elementType);
  // Reduction of input
  Value meanOfX = create.onnx.reduceMean(reductionType, X, axes);
  Value pow2OfMeanOfX = create.onnx.mul(meanOfX, meanOfX);
  Value XPow2 = create.onnx.mul(X, X);
  Value meanOfXPow2 = create.onnx.reduceMean(reductionType, XPow2, axes);
  Value var = create.onnx.sub(meanOfXPow2, pow2OfMeanOfX);
  Value varWithEpsilon = create.onnx.add(var, epsilon);
  Value stdDev = create.onnx.sqrt(varWithEpsilon);
  Value invStdDev = create.onnx.reciprocal(stdDev);
  Value d = create.onnx.sub(X, meanOfX);
  Value normalized = create.onnx.mul(d, invStdDev);
  Value Y = create.onnx.mul(normalized, lnOp.getScale());
  if (!isNoneValue(lnOp.getB()))
    Y = create.onnx.add(Y, lnOp.getB());
  replaceLayerNormalizationOp(rewriter, lnOp, Y, meanOfX, invStdDev);
  return success();
}

struct ONNXLayerNormalizationOpLowering
    : public OpConversionPattern<ONNXLayerNormalizationOp> {
  ONNXLayerNormalizationOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD)
      : OpConversionPattern(typeConverter, ctx), enableSIMD(enableSIMD) {}

  bool enableSIMD;

  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp lnOp,
      ONNXLayerNormalizationOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Get generic info.
    Operation *op = lnOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ONNXLayerNormalizationOp>(op);
    // Create builder and shape helper
    MDBuilder create(rewriter, loc);
    ONNXLayerNormalizationOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    MemRefType YMemRefType, meanMemRefType, ISDMemRefType;
    // bool computeMean = false, computeISD = false;

    // Get info.
    Value X = adaptor.getX();
    MemRefType XMemRefType = X.getType().cast<MemRefType>();
    Type elementType = XMemRefType.getElementType();
    int64_t XRank = XMemRefType.getRank();
    DimsExpr XDims;
    create.krnlIE.getShapeAsSymbols(X, XDims);
    int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
    int64_t innermostLoopCollapse = XRank - axis;

    // Detect if we can use SIMD
    int64_t VL = 0; // VL of 0 means no SIMD.
    int64_t estimatedSimdLoopTripCount;
    if (enableSIMD) {
      VectorMachineSupport *vms =
          VectorMachineSupport::getGlobalVectorMachineSupport();
      VL = create.vec.computeSuitableUnrollFactor(vms, XMemRefType, XDims,
          innermostLoopCollapse, 4, /*canPad*/ false,
          estimatedSimdLoopTripCount);
      LLVM_DEBUG({
        llvm::dbgs() << "  SIMD: " << innermostLoopCollapse << " loops, VL "
                     << VL << "\n";
        if (VL == 0)
          llvm::dbgs() << "  SIMD: no good VL\n";
      });
    }

#if 1
    return generateSIMDCode(rewriter, loc, lnOp, adaptor, shapeHelper, 1, 4);
#else
    return generateONNXLayerNormalizationOpONNXCode(rewriter, loc, lnOp);
#endif
  }

  using F1 = std::function<void(int64_t offsetInt, Value offsetVal)>;

  void inlineFor(MDBuilder &create, int64_t B, F1 genCode) const {
    for (int64_t offsetInt = 0; offsetInt < B; ++offsetInt) {
      Value offsetVal = create.math.constantIndex(offsetInt);
      genCode(offsetInt, offsetVal);
    }
  }

  void convertAlignAllocAndFlatten(MDBuilder &create, Value inputVal,
      DimsExpr &inputDims, int64_t axis, /*output*/ Value &memRef,
      /*output*/ Value &flatMemRef) const {
    // Convert input.
    Type convertedType = typeConverter->convertType(inputVal.getType());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    // Allocate.
    memRef = create.mem.alignedAlloc(memRefType, inputDims);
    // Flatten (do not keep flatten dims at this time).
    DimsExpr flatDims;
    flatMemRef = create.mem.reshapeToFlat2D(memRef, inputDims, flatDims, axis);
  }

  void generateIterWithSIMD(ConversionPatternRewriter &rewriter,
      MDBuilder &create, ONNXLayerNormalizationOp lnOp,
      /* flat inputs */ Value XMemRef,
      /* flat outputs */ Value YMemRef, Value meanMemRef, Value invStdDevMemRef,
      /* temps [B][vec] */ Value redMemRef, Value redMemRef2,
      /* value params */ Value i, Value redDim, Value epsilon,
      /* int params */ int64_t B, int64_t VL) const {
    // Vector type.
    Type elementType = YMemRef.getType().cast<ShapedType>().getElementType();
    VectorType vecType = VectorType::get({VL}, elementType);
    // Init the two reductions.
    Value init = create.math.constant(elementType, 0.0);
    Value initVec = create.vec.splat(vecType, init);
    Value zero = create.math.constantIndex(0);
    inlineFor(create, B, [&](int64_t d, Value o) {
      create.vec.store(initVec, redMemRef, {o, zero});
      create.vec.store(initVec, redMemRef2, {o, zero});
    });
#if 1
    // Perform reduction of entire vectors.
    IndexExpr izero = LiteralIndexExpr(0);
    IndexExpr iredDim = SymbolIndexExpr(redDim);
    create.affineKMem.forIE(izero, iredDim, VL,
        [&](onnx_mlir::AffineBuilderKrnlMem &ck, mlir::Value j) {
          MDBuilder create(ck);
          // load X, compute X**2, sum into reductions.
          inlineFor(create, B, [&](int64_t d, Value o) {
            Value ii = create.math.add(i, o);
            // Load X, compute X2.
            Value currX = create.vec.load(vecType, XMemRef, {ii, j});
            Value currXSquare = create.math.mul(currX, currX);
            // Load reductions.
            Value currRed = create.vec.load(vecType, redMemRef, {o, zero});
            Value currRed2 = create.vec.load(vecType, redMemRef2, {o, zero});
            // perform reductions.
            Value newRed = create.math.add(currRed, currX);
            Value newRed2 = create.math.add(currRed2, currXSquare);
            // Store reductions.
            create.vec.store(newRed, redMemRef, {o, zero});
            create.vec.store(newRed2, redMemRef2, {o, zero});
          });
        });

#else
    // Perform reduction of entire vectors.
    ValueRange reductionLoopDefs = create.krnl.defineLoops(1);
    ValueRange blockedReductionLoopDefs =
        create.krnl.block(reductionLoopDefs[0], VL);
    create.krnl.iterate({reductionLoopDefs[0]}, {blockedReductionLoopDefs[0]},
        {zero}, {redDim},
        [&](onnx_mlir::KrnlBuilder &ck, mlir::ValueRange indices) {
          MDBuilder create(ck);
          Value j = indices[0];
          // load X, compute X**2, sum into reductions.
          inlineFor(create, B, [&](int64_t d, Value o) {
            Value ii = create.math.add(i, o);
            // Load X, compute X2.
            Value currX = create.vec.load(vecType, XMemRef, {ii, j});
            Value currXSquare = create.math.mul(currX, currX);
            // Load reductions.
            Value currRed = create.vec.load(vecType, redMemRef, {o, zero});
            Value currRed2 = create.vec.load(vecType, redMemRef2, {o, zero});
            // perform reductions.
            Value newRed = create.math.add(currRed, currX);
            Value newRed2 = create.math.add(currRed2, currXSquare);
            // Store reductions.
            create.vec.store(newRed, redMemRef, {o, zero});
            create.vec.store(newRed2, redMemRef2, {o, zero});
          });
        });
#endif
    // Sum across, compute mean, var, standard deviation and its inverse.
    Value mean[B], invStdDev[B];
    Value redDimFloat = create.math.cast(elementType, redDim);
    Value oneFloat = create.math.constant(elementType, 1.0);
    inlineFor(create, B, [&](int64_t d, Value o) {
      // Load reductions.
      Value finalRed = create.vec.load(vecType, redMemRef, {o, zero});
      Value finalRed2 = create.vec.load(vecType, redMemRef2, {o, zero});
      // Horizontal reductions.
      Value currSum =
          create.vec.reduction(VectorBuilder::CombiningKind::ADD, finalRed);
      Value currSum2 =
          create.vec.reduction(VectorBuilder::CombiningKind::ADD, finalRed2);
      // Compute means.
      mean[d] = create.math.div(currSum, redDimFloat);
      Value mean2 = create.math.div(currSum2, redDimFloat);
      // Compute standard deviation (with epsilon) and its inverse.
      Value meanSquare = create.math.mul(mean[d], mean[d]);
      Value var = create.math.sub(mean2, meanSquare);
      Value varEps = create.math.add(var, epsilon);
      Value stdDev = create.math.sqrt(varEps);
      invStdDev[d] = create.math.div(oneFloat, stdDev);
    });
// Normalize of entire vectors.
#if 1
    create.affineKMem.forIE(izero, iredDim, VL,
        [&](onnx_mlir::AffineBuilderKrnlMem &ck, mlir::Value j) {
          MDBuilder create(ck);
          // load X, compute X**2, sum into reductions.
          inlineFor(create, B, [&](int64_t d, Value o) {
            Value ii = create.math.add(i, o);
            // Load X, compute X2.
            Value currX = create.vec.load(vecType, XMemRef, {ii, j});
            Value meanSplat = create.vec.splat(vecType, mean[d]);
            Value XMinusMean = create.math.sub(currX, meanSplat);
            Value invStdDevSplat = create.vec.splat(vecType, invStdDev[d]);
            Value normalizedX = create.math.mul(XMinusMean, invStdDevSplat);
            // Skip for now the scale and bias.
            create.vec.store(normalizedX, YMemRef, {ii, j});
          });
        });

#else
    ValueRange reductionLoopDefs2 = create.krnl.defineLoops(1);
    ValueRange blockedReductionLoopDefs2 =
        create.krnl.block(reductionLoopDefs2[0], VL);
    create.krnl.iterate({reductionLoopDefs2[0]}, {blockedReductionLoopDefs2[0]},
        {zero}, {redDim},
        [&](onnx_mlir::KrnlBuilder &ck, mlir::ValueRange indices) {
          MDBuilder create(ck);
          Value j = indices[0];
          // load X, compute X**2, sum into reductions.
          inlineFor(create, B, [&](int64_t d, Value o) {
            Value ii = create.math.add(i, o);
            // Load X, compute X2.
            Value currX = create.vec.load(vecType, XMemRef, {ii, j});
            Value meanSplat = create.vec.splat(vecType, mean[d]);
            Value XMinusMean = create.math.sub(currX, meanSplat);
            Value invStdDevSplat = create.vec.splat(vecType, invStdDev[d]);
            Value normalizedX = create.math.mul(XMinusMean, invStdDevSplat);
            // Skip for now the scale and bias.
            create.vec.store(normalizedX, YMemRef, {ii, j});
          });
        });
#endif
    // save mean and std dev if requested.
    if (meanMemRef) {
      inlineFor(create, B, [&](int64_t d, Value o) {
        Value ii = create.math.add(i, o);
        create.krnl.store(mean[d], meanMemRef, {ii, zero});
      });
    }
    if (invStdDevMemRef) {
      inlineFor(create, B, [&](int64_t d, Value o) {
        Value ii = create.math.add(i, o);
        create.krnl.store(invStdDev[d], invStdDevMemRef, {ii, zero});
      });
    }
  }

  LogicalResult generateSIMDCode(ConversionPatternRewriter &rewriter,
      Location loc, ONNXLayerNormalizationOp lnOp,
      ONNXLayerNormalizationOpAdaptor &adaptor,
      ONNXLayerNormalizationOpShapeHelper &shapeHelper, int64_t B,
      int64_t VL) const {
    MDBuilder create(rewriter, loc);
    Value XMemRef = adaptor.getX();
    MemRefType XMemRefType = XMemRef.getType().cast<MemRefType>();
    Type elementType = XMemRefType.getElementType();
    int64_t XRank = XMemRefType.getRank();
    int64_t axis = getAxisInRange(lnOp.getAxis(), XRank);
    // Get epsilon as a scalar.
    Value epsilon =
        create.math.constant(elementType, lnOp.getEpsilon().convertToDouble());

    // Flatten inputs.
    Value XFlatMemRef, scaleFlatMemRef, BFlatMemRef;
    DimsExpr XFlatDims;
    XFlatMemRef = create.mem.reshapeToFlat2D(
        XMemRef, shapeHelper.inputsDims[0], XFlatDims, axis);

    // Convert outputs, alloc data, and flatten them too.
    Value YMemRef, meanMemRef, invStdDevMemRef;
    Value YFlatMemRef, meanFlatMemRef, invStdDevFlatMemRef;
    convertAlignAllocAndFlatten(create, lnOp.getY(),
        shapeHelper.getOutputDims(0), axis, YMemRef, YFlatMemRef);
    if (!isNoneValue(lnOp.getMean()))
      convertAlignAllocAndFlatten(create, lnOp.getMean(),
          shapeHelper.getOutputDims(1), axis, meanMemRef, meanFlatMemRef);
    if (!isNoneValue(lnOp.getInvStdDev()))
      convertAlignAllocAndFlatten(create, lnOp.getInvStdDev(),
          shapeHelper.getOutputDims(2), axis, invStdDevMemRef,
          invStdDevFlatMemRef);
    // Alloc mem for reductions (should be private if parallel)
    MemRefType tmpRedType = MemRefType::get({B, VL}, elementType);
    Value tmpRedMemRef = create.mem.alignedAlloca(tmpRedType);
    Value tmpRedMemRef2 = create.mem.alignedAlloca(tmpRedType);
#if 0
    ValueRange loopDefs = create.krnl.defineLoops(1);
    IndexExpr zero = LiteralIndexExpr(0);
    create.krnl.iterateIE({loopDefs[0]}, {loopDefs[0]}, {zero}, {XFlatDims[0]},
        [&](KrnlBuilder &ck, ValueRange loopIndices) {
          MDBuilder create(ck);
          generateIterWithSIMD(rewriter, create, lnOp, XFlatMemRef, YFlatMemRef,
              meanFlatMemRef, invStdDevFlatMemRef, tmpRedMemRef, tmpRedMemRef2,
              loopIndices[0], XFlatDims[1].getValue(), epsilon, 1, VL);
        });
#else
    // Iterate over 1st dim by block
    ValueRange loopDefs = create.krnl.defineLoops(1);
    ValueRange blockedLoopDef = create.krnl.block(loopDefs[0], B);
    IndexExpr zero = LiteralIndexExpr(0);
    create.krnl.iterateIE({loopDefs[0]}, {blockedLoopDef[0]}, {zero},
        {XFlatDims[0]}, [&](KrnlBuilder &ck, ValueRange blockedLoopIndices) {
          MDBuilder create(ck);
#if 0
          IndexExprScope innerScope(ck);
          IndexExpr blockedCurrIndex = DimIndexExpr(blockedLoopIndices[0]);
          IndexExpr blockedUB =
              SymbolIndexExpr(XFlatDims[0].getValue()); // hi alex, take value?
          IndexExpr isFull = create.krnlIE.isTileFull(
              blockedCurrIndex, LiteralIndexExpr(B), blockedUB);
          Value zero = create.math.constantIndex(0);
          Value isNotFullVal = create.math.slt(isFull.getValue(), zero);
#endif
          // hi alex, only do full
          IndexExprScope innerScope(ck);
          generateIterWithSIMD(rewriter, create, lnOp, XFlatMemRef, YFlatMemRef,
              meanFlatMemRef, invStdDevFlatMemRef, tmpRedMemRef, tmpRedMemRef2,
              blockedLoopIndices[0], XFlatDims[1].getValue(), epsilon, B, VL);
        });
    // if block full
    // if block not full... iterate over individual blocks
#endif
    // hi alex, not sure if I can just return the non-flatten variables, or if I
    // need to reshape fhe flattened ones.
    replaceLayerNormalizationOp(
        rewriter, lnOp, YMemRef, meanMemRef, invStdDevMemRef);
    return success();
  }
};

void populateLoweringONNXNormalizationOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableSIMD) {
  patterns.insert<ONNXBatchNormalizationInferenceModeOpLowering>(
      typeConverter, ctx);
  patterns.insert<ONNXInstanceNormalizationOpLowering>(typeConverter, ctx);
  patterns.insert<ONNXLayerNormalizationOpLowering>(
      typeConverter, ctx, enableSIMD);
}

} // namespace onnx_mlir
