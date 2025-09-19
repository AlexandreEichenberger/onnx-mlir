/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- ProcessStickData.cpp - Process Stick data ----------------===//
//
// Copyright 2024-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ZHigh operations to Krnl/Affine/SCF
// operations that operates on stickified input/output data.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.hpp"
#include "src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/DialectBuilder.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/Quantization/QuantizeHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/SmallVectorHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

// True: optimize with write of 8 values for the last iter of a stick,
// regardless if we need each of the results. False, conservatively only write
// the "allowed" values in the output for the last couple of values.
#define STICK_OUTPUT_WRITE_PAST_BOUNDS true

// Include necessary info from elementwise so as to gen code here.
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"

//===----------------------------------------------------------------------===//
// Handle quantization on stickified inputs
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
// Implementation of quantize helper function.
void emitDynamicQuantizationLinearMinMaxFromStickifiedInput(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Value input, StringAttr inputLayout, Value &inputMin, Value &inputMax,
    bool enableSIMD, bool enableParallel) {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MathBuilder, MemRefBuilder, VectorBuilder>;
  MDBuilder create(rewriter, loc);

  // Extract dims from input, set lbs/ubs.
  DimsExpr dims;
  create.krnlIE.getShapeAsSymbols(input, dims);
  int64_t rank = dims.size();
  IndexExpr zero = LitIE(0);
  DimsExpr lbs(rank, zero);
  DimsExpr ubs = dims;

  // Decide parameters.
  // UnrollVL decides how many vectors of 8 DLF16 will be processed at once.
  int64_t unrollVL = 4; // Experimentally good unroll factor.
  int64_t archVL = 8;   // DLF16.
  int64_t totVL = unrollVL * archVL;

  // If not parallel, threadNum = 1, forExplicitParallelLoopIE will simply pass
  // through the lb/ub, so ok to have parID = 0 for the sequential cases.
  int64_t parId = 0;
  int64_t threadNum = 1;
  if (enableParallel) {
    int64_t parId = tryCreateKrnlParallel(create.krnl, op,
        "simd min/max for DQL in parallel", {}, lbs, ubs, 0, rank - 1, {},
        /*min iter for going parallel*/ 8, /*createKrnlParallel=*/false);
    if (parId == -1) {
      enableParallel = false;
    } else {
      threadNum = 8; // TODO use more flexible value.
    }
  }

  // Alloc temp buffers (more when using parallel).
  Type f32Type = rewriter.getF32Type();
  // For each thread, we can use totVL temp values for the current min/max.
  // But to increase the compute ratio over mem, we will reuse the same tmp
  // memory location for a pair of totVL values being processed.
  int64_t tmpSizePerThread = totVL / 2; // Reduce pair in same tmp.
  int64_t tmpSize = threadNum * tmpSizePerThread;
  MemRefType redType = MemRefType::get({tmpSize}, f32Type);
  VectorType vec8xF32Type = VectorType::get({archVL}, f32Type);
  VectorType vec4xF32Type = VectorType::get({archVL / 2}, f32Type);

  Value minTmp = create.mem.alignedAlloc(redType);
  Value maxTmp = create.mem.alignedAlloc(redType);

  // Init min and max.
  Value minInit = create.math.positiveInf(f32Type);
  Value splatMinInit = create.vec.splat(vec8xF32Type, minInit);
  Value maxInit = create.math.negativeInf(f32Type);
  Value splatMaxInit = create.vec.splat(vec8xF32Type, maxInit);
  // Could parallelize init, here main thread do it all. Use SIMD of 8x.
  for (int64_t u = 0; u < tmpSize; u += 8) {
    IndexExpr offset = LitIE(u);
    create.vec.storeIE(splatMinInit, minTmp, {offset});
    create.vec.storeIE(splatMaxInit, maxTmp, {offset});
  }

  // Reduction into these temps.
  IndexExpr tNum = LitIE(threadNum);
  create.krnl.forExplicitParallelLoopIE(lbs[parId], ubs[parId], tNum,
      [&](const KrnlBuilder &ck, ValueRange loopInd) {
        IndexExprScope scope(ck);
        IndexExpr t = DimIE(loopInd[0]);
        DimsExpr currDims = DimListIE(dims);
        // Reduce lbs, ubs for parallel region, if any.
        DimsExpr currLbs = DimListIE(lbs);
        DimsExpr currUbs = DimListIE(ubs);
        // In sequential cases (threadNum ==1, loopInd[1,2]== orig lb,ub).
        currLbs[parId] = SymIE(loopInd[1]);
        currUbs[parId] = SymIE(loopInd[2]);
        // Cannot use krnl because we may not have affine bounds.
        SCFBuilder sb(ck);
        IterateOverStickInputData<SCFBuilder>(
            sb, op, currLbs, currUbs, currDims, inputLayout, input, nullptr,
            unrollVL, /*enableParallel*/ false,
            /*prefetch, disable as it causes issue with affine*/ false,
            [&](const KrnlBuilder &b, SmallVectorImpl<Value> &vecOf4xF32Vals,
                DimsExpr &loopIndices) {
              MDBuilder create(b);
              int64_t size = vecOf4xF32Vals.size();
              assert((size == 2 || size == 2 * unrollVL) && "unexpected size");
              // Since all threads share the same tmpMin/Max, needs to offset by
              // t * <size for one thread>.
              IndexExpr threadOffset = SymIE(t) * tmpSizePerThread;
              size = size / 2; // handle pairs of 2, so size=1 or unrollVL.
              for (int i = 0; i < size; ++i) {
                Value val0 = vecOf4xF32Vals[2 * i];
                Value val1 = vecOf4xF32Vals[2 * i + 1];
                // Load appropriate tmp, compute min/max, store in tmp.
                IndexExpr offset = threadOffset + LitIE(4 * i);
                Value currMin =
                    create.vec.loadIE(vec4xF32Type, minTmp, {offset});
                Value currMax =
                    create.vec.loadIE(vec4xF32Type, maxTmp, {offset});
                currMin = create.math.min(currMin, val0);
                currMax = create.math.max(currMax, val0);
                currMin = create.math.min(currMin, val1);
                currMax = create.math.max(currMax, val1);
                create.vec.storeIE(currMin, minTmp, {offset});
                create.vec.storeIE(currMax, maxTmp, {offset});
              }
            },
            [&](const KrnlBuilder &b, Value scalarF32Val,
                DimsExpr &loopIndices) {
              MDBuilder create(b);
              Value currMin = create.krnl.loadIE(minTmp, {zero});
              Value currMax = create.krnl.loadIE(maxTmp, {zero});
              currMin = create.math.min(currMin, scalarF32Val);
              currMax = create.math.max(currMax, scalarF32Val);
              create.krnl.storeIE(currMin, minTmp, {zero});
              create.krnl.storeIE(currMax, maxTmp, {zero});
            }); // Iterate over stick.
      });       // Explicit parallel loop (sequential if threadNum==1).

  // Now we have all the partial min/max inside the minTmp/maxTmp: reduce each
  // vectors with each others. Main thread reduces all the values. Use SIMD of
  // 8x.
  Value finalVecMin = create.vec.loadIE(vec8xF32Type, minTmp, {zero});
  Value finalVecMax = create.vec.loadIE(vec8xF32Type, maxTmp, {zero});
  for (int u = 8; u < tmpSize; u += 8) {
    IndexExpr offset = LitIE(u);
    Value currMin = create.vec.loadIE(vec8xF32Type, minTmp, {offset});
    Value currMax = create.vec.loadIE(vec8xF32Type, maxTmp, {offset});
    finalVecMin = create.math.min(finalVecMin, currMin);
    finalVecMax = create.math.max(finalVecMax, currMax);
  }

  // Horizontal reduction of the vectors into a scalar.
  inputMin = create.vec.reduction(VectorBuilder::MIN, finalVecMin);
  inputMax = create.vec.reduction(VectorBuilder::MAX, finalVecMax);
}

//===----------------------------------------------------------------------===//
// Handle elementwise operations with stickified inputs/outputs
//===----------------------------------------------------------------------===//

using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
    MemRefBuilder, VectorBuilder, SCFBuilder, MathBuilder, ZLowBuilder>;

using MultiValuesOfF32IterateBodyFn = std::function<mlir::Value(
    const KrnlBuilder &b, mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals)>;

//===----------------------------------------------------------------------===//
// High level support with StickComputeSupport.

StickComputeSupport::StickComputeSupport(MDBuilder &create, Operation *op,
    ValueRange originalInputMemRef, Value originalOutputMemRef,
    MultiValuesOfF32IterateBodyFn processVectorOfF32Vals,
    bool disableSaturation)
    : processVectorOfF32Vals(processVectorOfF32Vals),
      disableSaturation(disableSaturation) {
  // Init global constants.
  litZero = LitIE(0);
  lit2 = LitIE(2);
  litStickLen = LitIE(stickLen);
  Type f16Type = create.getBuilder().getF16Type();
  vecF16Type = VectorType::get({archVL}, f16Type);
  Type f32Type = create.getBuilder().getF32Type();
  vecF32Type = VectorType::get({archVL / 2}, f32Type);

  // Register inputs.
  // 1) Add input operands into the io lists.
  inputNum = op->getNumOperands();
  ioNum = inputNum + 1;
  for (int64_t io = 0; io < inputNum; ++io) {
    ioOriginalOper.emplace_back(op->getOperand(io));
    ioOriginalMemRef.emplace_back(originalInputMemRef[io]);
  }
  // 2) Add single output (last entry) in io lists.
  Value originalOutput = op->getResult(0);
  ioOriginalOper.emplace_back(originalOutput);
  ioOriginalMemRef.emplace_back(originalOutputMemRef);
  int64_t innermostOutputShape = getShape(originalOutput.getType(), -1);

  // Initialize the computed io values.
  ioMemRef = ioOriginalMemRef; // Initially the two are the same.
  ioIsStick = mlir::BitVector(ioNum, false);
  ioIsBroadcast = mlir::BitVector(ioNum, false);
  ioIsBuffer = mlir::BitVector(ioNum, false);

  // Iterate over all inputs & output to assign the computed io values.
  for (int io = 0; io < ioNum; ++io) {
    Value originalVal = ioOriginalOper[io];
    Value originalMemRef = ioOriginalMemRef[io];
    Type originalType = originalVal.getType();
    int64_t innermostShape = getShape(originalType, -1);
    auto originalShape = getShape(originalType);
    ioIsBroadcast[io] = (innermostShape == 1 && innermostOutputShape != 1);
    ioIsStick[io] = zhigh::isZTensor(originalType);
    ioIsBuffer[io] = originalShape.size() == 2 && originalShape[0] == 1 &&
                     originalShape[1] == archVL;
    if (ioIsStick[io] && !ioIsBroadcast[io]) {
      // Set in ioMemRef the flattened [2, 64] view.
      assert(
          zhigh::supportedLayoutForCompilerGeneratedStickUnstick(originalVal) &&
          "unsupported layout");
      DimsExpr castShape = {lit2, litStickLen};
      ioMemRef[io] = create.mem.reinterpretCast(originalMemRef, castShape);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  " << io << ": " << (ioIsStick[io] ? "is stick " : "")
               << (ioIsBroadcast[io] ? "is broadcast " : "")
               << (ioIsBuffer[io] ? "is buffer " : "") << "operand\n");
  }
  assert(!ioIsBroadcast[/* output index*/ inputNum] &&
         "expect no broadcasted output");
}

// Pick the innermost loop indices as needed by the rank of val to compute its
// access function. Make sure that when the shape is of size 1, we use index 0
// regardless of the values of loopIndices to facilitate broadcast (compile
// time). Also add inner offset the the innermost dimension of the resulting
// access function.
// Inputs:
//   val: original tensor value, only use the rank and its shape.
//   loopIndices: will use the innermost ones (depending on rank of val).
//   additionalInnerOffset: add to the innermost dim (offset).
DimsExpr StickComputeSupport::computeAccessFct(
    Value val, DimsExpr &loopIndices, IndexExpr additionalInnerOffset) {
  DimsExpr accessFct;
  int64_t valRank = getRank(val.getType());
  int64_t loopRank = loopIndices.size();
  accessFct.clear();
  for (int i = 0; i < valRank; ++i) {
    if (getShape(val.getType(), i) == 1) {
      // Possible broadcast situation, ignore loop index, just use 0.
      accessFct.emplace_back(LitIE(0));
    } else {
      // Use innermost loop indices.
      IndexExpr index = loopIndices[loopRank - valRank + i];
      if (i == valRank - 1)
        index = index + additionalInnerOffset; // Add innermost offset.
      accessFct.emplace_back(index);
    }
  }
  return accessFct;
}

void StickComputeSupport::prepareInsideTiledLoop(
    MDBuilder &create, DimsExpr &tiledOuterIndices, IndexExpr E1) {
  int64_t rank = tiledOuterIndices.size();
  int64_t d1 = rank - 1;

  // Initialize data
  ioStickOffsets = DimsExpr(ioNum, nullptr);
  inputHigh = SmallVector<Value, 4>(inputNum, nullptr);
  inputLow = SmallVector<Value, 4>(inputNum, nullptr);
  // Tiled outer indices: innermost iterates through sticks (multiple of 64
  // values). Generate here the original (untiled) indices as well.
  outerIndices = tiledOuterIndices;
  outerIndices[d1] = tiledOuterIndices[d1] * litStickLen;

  // Iterate over all input and the output.
  for (int64_t io = 0; io < ioNum; io++) {
    if (ioIsBroadcast[io]) {
      // Broadcast of stick/scalar: load scalar value.
      // Note that while the innermost dim e1 maybe large, to broadcast,
      // we must have the innermost shape being 1, and thus
      // computeAccessFct will force that innermost access function to 0.
      Value memref = ioMemRef[io];
      DimsExpr accessFct = computeAccessFct(memref, outerIndices, litZero);
      Value scalar = create.krnl.loadIE(memref, accessFct);
      if (ioIsStick[io]) {
        Value vecF16 = create.vec.splat(vecF16Type, scalar);
        Value vecHigh, vecLow;
        create.zlow.convertDLF16ToF32(vecF16, vecHigh, vecLow);
        inputHigh[io] = inputLow[io] =
            vecHigh; // Since splatted, low and high are the same.
      } else {
        inputHigh[io] = inputLow[io] = create.vec.splat(vecF32Type, scalar);
      }
    } else if (ioIsStick[io]) {
      // Stick data that is not broadcasted: need SIMD support.
      // Translate the tile index to the stick offset. Have to
      // give the actual indices, not the tiled ones.
      Value originalMemRef = ioOriginalMemRef[io];
      DimsExpr accessFct =
          computeAccessFct(originalMemRef, outerIndices, litZero);
      Value offset =
          create.krnl.getLinearOffsetIndexIE(originalMemRef, accessFct);
      ioStickOffsets[io] = DimIE(offset).floorDiv(litStickLen);
    }
  }
}

// LoopIndices in the context of the innermost loop.
void StickComputeSupport::loadVector(MDBuilder &create,
    DimsExpr &localOuterIndices, IndexExpr l, int64_t u, int64_t i) {
  // Broadcast loaded inside the outer loops.
  if (ioIsBroadcast[i])
    return;
  // Compute innermost offset (l: index of loop, u:  unrolling by archVL).
  DimsExpr indices;
  IndexExpr stickOffset, offset;
  if (ioIsBuffer[i]) {
    // For buffer, all accesses are zero.
    indices = {litZero, litZero};
    stickOffset = litZero;
    offset = litZero;
  } else {
    indices = localOuterIndices;
    stickOffset = ioStickOffsets[i];
    offset = l + (archVL * u);
  }
  if (ioIsStick[i]) {
    DimsExpr accessFct = {DimIE(stickOffset), offset};
    Value vecOfDLF16 = create.vec.loadIE(vecF16Type, ioMemRef[i], accessFct);
    create.zlow.convertDLF16ToF32(vecOfDLF16, inputHigh[i], inputLow[i]);
    fprintf(stderr, "hi alex from stick load\n");
    inputHigh[i].dump();
    inputLow[i].dump();
  } else {
    DimsExpr accessFct = computeAccessFct(ioMemRef[i], indices, offset);
    inputHigh[i] = create.vec.loadIE(vecF32Type, ioMemRef[i], accessFct);
    Value lowOffset = create.math.constantIndex(archVL / 2);
    inputLow[i] =
        create.vec.loadIE(vecF32Type, ioMemRef[i], accessFct, {lowOffset});
  }
}

void StickComputeSupport::storeVector(MDBuilder &create,
    DimsExpr &localOuterIndices, IndexExpr l, int64_t u, int64_t o,
    Value tempBufferMemRef, Value outputHigh, Value outputLow) {
  assert(!ioIsBroadcast[o] && "output should not be broadcast");
  // Compute innermost offset (l: index of loop, u:  unrolling by archVL).
  DimsExpr indices;
  IndexExpr offset, stickOffset;
  Value memref;
  if (ioIsBuffer[o] || tempBufferMemRef) {
    // For buffer, all accesses are zero.
    indices = {litZero, litZero};
    stickOffset = litZero;
    offset = litZero;
    memref = tempBufferMemRef ? tempBufferMemRef : ioMemRef[o];
  } else {
    indices = localOuterIndices;
    stickOffset = ioStickOffsets[o];
    offset = l + (archVL * u);
    memref = ioMemRef[o];
  }
  if (ioIsStick[o]) {
    Value dlf16 =
        create.zlow.convertF32ToDLF16(outputHigh, outputLow, disableSaturation);
    DimsExpr accessFct = {DimIE(stickOffset), offset};
    create.vec.storeIE(dlf16, memref, accessFct);
  } else {
    DimsExpr accessFct = computeAccessFct(memref, indices, offset);
    create.vec.storeIE(outputHigh, memref, accessFct);
    Value lowOffset = create.math.constantIndex(archVL / 2);
    create.vec.storeIE(outputLow, memref, accessFct, {lowOffset});
  }
}

void StickComputeSupport::loadComputeStore(
    MDBuilder &create, IndexExpr l, int64_t u, Value tempBufferMemRef) {
  DimsExpr localOuterIndices = DimListIE(outerIndices);
  // Load inputs (all but for broadcast, done in prepareInsideTiledLoop).
  for (int64_t i = 0; i < inputNum; ++i) {
    loadVector(create, localOuterIndices, l, u, i);
    assert(inputHigh[i] != nullptr && "expected defined input high");
    assert(inputLow[i] != nullptr && "expected defined input low");
  }
  // Compute high/low in f32 format.
  Value outputHigh = processVectorOfF32Vals(create.krnl, inputHigh);
  Value outputLow = processVectorOfF32Vals(create.krnl, inputLow);
  // Store results.
  storeVector(create, localOuterIndices, l, u, /* index */ inputNum,
      tempBufferMemRef, outputHigh, outputLow);
}

//===----------------------------------------------------------------------===//
// Process operation with stick inputs/output.

static void IterateOverStickInputOutput(const KrnlBuilder &b, Operation *op,
    ValueRange operands /*converted*/, Value alloc, DimsExpr &outputDims,
    int64_t unrollVL, bool enableParallel, bool disableSaturation,
    bool enablePrefetch, MultiValuesOfF32IterateBodyFn processVectorOfF32Vals) {

  // Init builder and scopes.
  MDBuilder create(b);
  // IndexExprScope initialScope(b);
  //  Get info and check some inputs.
  int64_t rank = outputDims.size();
  int64_t d1 = rank - 1;
  IndexExpr E1 = outputDims[d1];
  assert(op->getNumResults() == 1 && "handle only 1 output ops");

  int64_t archVL = StickComputeSupport::archVL;
  int64_t stickLen = StickComputeSupport::stickLen;
  int64_t totVL = archVL * unrollVL;
  assert(stickLen % totVL == 0 && "bad unrollVL factor");
  IndexExpr litZero = LitIE(0);
  IndexExpr litStickLen = LitIE(stickLen);

  // Create loop iterations. We iterate over E1 as sticks of 64 elements. Lbs
  // and ubs reflect the iteration over the sticks (tiled data points).
  DimsExpr tiledLbs(rank, litZero);
  DimsExpr tiledUbs = outputDims;
  tiledUbs[d1] = E1.ceilDiv(litStickLen);

  // Parallel... Should not be turned on when parallelized in the outside.
  int64_t parId = 0;
  if (enableParallel) {
    // TODO: may want to check if ub of rank makes sense here.
    // Its ok here even to partition rank-1, included in (0..rank(, because
    // rank-1 is tiled. So we are still dealing with multiple of sticks.
    parId = tryCreateKrnlParallel(create.krnl, op,
        "compiler-generated stickify", {}, tiledLbs, tiledUbs, 0, rank, {},
        /*min iter for going parallel*/ 8, /*createKrnlParallel=*/false);
    if (parId == -1)
      enableParallel = false;
  }

  StickComputeSupport stickCS(
      create, op, operands, alloc, processVectorOfF32Vals, disableSaturation);

  // Predicates used to avoid creating code that is never used.
  bool neverHas64 = E1.isLiteralAndSmallerThan(stickLen);
  bool neverHas8 = E1.isLiteralAndSmallerThan(archVL);
  bool hasOnly64 = E1.isLiteral() && (E1.getLiteral() % stickLen == 0);
  bool hasOnly8 = E1.isLiteral() && (E1.getLiteral() % archVL == 0);

  if (STICK_OUTPUT_WRITE_PAST_BOUNDS && stickCS.isStickifiedOutput()) {
    // Output is stickified, we can write 8 values for the last iteration no
    // mater what, possibly over-writing values that we are not supposed to
    // write, but we know the memory exists as we always allocate a stick of 64
    // values.
    neverHas8 = false; // Force at least one iteration into the 8-way simd loop.
    hasOnly8 = true;   // Skip the scalar loop with the custom buffer.
  }
  LLVM_DEBUG(
      llvm::dbgs() << "  Predicates: " << (neverHas64 ? "never-has-64 " : "")
                   << (neverHas8 ? "never-has-8 " : "")
                   << (hasOnly64 ? "has-only-64 " : "")
                   << (hasOnly8 ? "has-only-8\n" : "\n"));

  // Iterates over sticks.
  llvm::SmallVector<int64_t, 4> steps(rank, 1);
  llvm::SmallVector<bool, 4> useParallel(rank, false);
  if (enableParallel)
    useParallel[parId] = true;
  b.forLoopsIE(tiledLbs, tiledUbs, steps, useParallel,
      [&](const KrnlBuilder &b, mlir::ValueRange tiledLoopInd) {
        IndexExprScope outerScope(b);
        MDBuilder create(b);
        DimsExpr tiledOuterIndices = DimListIE(tiledLoopInd);
        // Computation for accessing data (not tiled, actual indices).
        DimsExpr outerIndices = tiledOuterIndices;
        IndexExpr E1 = DimIE(outputDims[d1]); // Original upper bound in d1.
        IndexExpr e1 = outerIndices[d1] = tiledOuterIndices[d1] * litStickLen;

        stickCS.prepareInsideTiledLoop(create, tiledOuterIndices, E1);

        if (enablePrefetch) {
          // TODO: enable prefetch
          // Prefetch all in ioMemRefValues
          // create.krnl.prefetchIE(input, outerIndices, /*write*/ false,
          //    /*locality*/ 1);
        }
        // Check if we have a full stick (aka end of stick is not beyond UB).
        IndexExpr hasFullStick;
        if (hasOnly64) {
          hasFullStick = PredIE(true); // Has only full sicks.
        } else if (neverHas64) {
          hasFullStick = PredIE(false); // Doesn't even has 1 stick.
        } else {
          IndexExpr isFull = create.krnlIE.isTileFull(e1, litStickLen, E1);
          hasFullStick = (isFull >= 0);
        }
        create.scf.ifThenElse(
            hasFullStick.getValue(),
            // If is full, process all 64 values here.
            [&](const SCFBuilder b) {
              if (neverHas64)
                return; // Nothing to do here. Avoid generating dead code.
              MDBuilder create(b);
              // Iterate through stick by totVL (aka 8 * unroll).
              create.scf.forLoopIE(litZero, litStickLen, totVL, /*par*/ false,
                  [&](const SCFBuilder b, mlir::ValueRange loopInd) {
                    IndexExprScope innerScope(b, &outerScope);
                    MDBuilder create(b);
                    IndexExpr l = DimIE(loopInd[0]);
                    DimsExpr innerIndices = DimListIE(outerIndices);
                    for (int64_t u = 0; u < unrollVL; ++u)
                      stickCS.loadComputeStore(create, l, u);
                  });
            },
            // Else, we don't have a full (64 e1) tile.
            [&](SCFBuilder b) {
              if (hasOnly64)
                return; // Do not generate dead code.
              MDBuilder create(b);
              IndexExprScope middleScope(b, &outerScope);
              IndexExpr tripCount = DimIE(E1) - DimIE(e1);
              if (!neverHas8) {
                // Not full 64, process all sub-tiles of 8 values here.
                // Note: if we only have multiple of VL, loop below will
                // handle all VL-full as we subtract (VL-1). Aka if VL=8 and
                // tripCount = 16, tripCountSimdByVL is 16 - 7 = 9.
                // Thus we iterate over i=0 & i=8 as both are < 9.
                int64_t correction = archVL - 1;
                if (STICK_OUTPUT_WRITE_PAST_BOUNDS &&
                    stickCS.isStickifiedOutput()) {
                  // Overwrite is allowed, so if VL=8 and trip count = 16: will
                  // execute i=0 and i=8 (both full). But if trip count = 17,
                  // then will execute i=0 & 8 (full), and i=16 (to compute/save
                  // the stick[x,16] single value, but overriding
                  // stick[x, 17..23] with garbage values).
                  correction = 0;
                }
                IndexExpr tripCountSimdByVL = tripCount - correction;
                create.scf.forLoopIE(litZero, tripCountSimdByVL, archVL,
                    /*par*/ false, [&](SCFBuilder b, mlir::ValueRange loopInd) {
                      IndexExprScope innerScope(b, &middleScope);
                      MDBuilder create(b);
                      IndexExpr l = DimIE(loopInd[0]);
                      DimsExpr innerIndices = DimListIE(outerIndices);
                      stickCS.loadComputeStore(create, l, 0);
                    });
              }
              if (!hasOnly8) {
                // Deal with the last <8 values: compute f32 using simd.
                // IndexExpr remainingScalarValues = tripCount % archVL;

                // Can use E1 instead of trip count as trip count substract
                // multiple of 64 to E1, and 64 % (archVal=8) = 0.
                IndexExpr remainingScalarValues = DimIE(E1) % archVL;
                IndexExpr lastL = tripCount - remainingScalarValues;
                IndexExpr innerIndexPlusLastL = DimIE(e1) + lastL;
                // Need a buffer to store partial results (less than 8) for last
                // iterations. Use a type of [1][8] so that it can contain up to
                // 7 partial results. Use a unit first dim to match the rank of
                // the reinterpreted casts.
                Type outputElementType =
                    getElementType(op->getResult(0).getType());
                MemRefType bufferType =
                    mlir::MemRefType::get({1, archVL}, outputElementType);
                Value outputBuffer = create.mem.alignedAlloc(bufferType);

                // Compute results and store into output buffer.
                // Buffer holds original or stickified (normalized) results
                // depending on the output type.
                DimsExpr innerIndices = DimListIE(outerIndices);
                stickCS.loadComputeStore(create, litZero, 0, outputBuffer);
                // Scalar store of buffer values.
                create.scf.forLoopIE(litZero, remainingScalarValues, 1,
                    /*par*/ false, [&](SCFBuilder b, mlir::ValueRange loopInd) {
                      IndexExprScope innerScope(b, &middleScope);
                      MDBuilder create(b);
                      IndexExpr l = DimIE(loopInd[0]);
                      Value bufferVal =
                          create.krnl.loadIE(outputBuffer, {litZero, l});
                      // Even if stickified, we don't need simd store, and thus
                      // we can use the memref format without the view.
                      DimsExpr innerIndices = DimListIE(outerIndices);
                      innerIndices[d1] = DimIE(innerIndexPlusLastL) + l;
                      create.krnl.storeIE(bufferVal, alloc, innerIndices);
                    });
              }
            });
      });
}

// Check that all the input/outputs are float32 without ztensor, or dlf16 with
// tensor. Must have at least one dlf16 to return true.
static bool isZTensorOfF32AndDLF16(Operation *op) {
  bool hasDLF16 = false;
  for (Value val : op->getOperands()) {
    Type elementType = getElementType(val.getType());
    if (zhigh::isZTensor(val.getType())) {
      if (!elementType.isF16()) {
        return false;
      }
      hasDLF16 = true;
    } else {
      if (!elementType.isF32()) {
        return false;
      }
    }
  }
  for (Value val : op->getResults()) {
    Type elementType = getElementType(val.getType());
    if (zhigh::isZTensor(val.getType())) {
      if (!elementType.isF16()) {
        return false;
      }
      hasDLF16 = true;
    } else {
      if (!elementType.isF32()) {
        return false;
      }
    }
  }
  return hasDLF16;
}

//===----------------------------------------------------------------------===//
// Elementwise patterns.

template <typename ElementwiseOp>
struct ONNXElementwiseOpLoweringWithNNPALayout
    : public OpConversionPattern<ElementwiseOp> {
  using OpAdaptor = typename ElementwiseOp::Adaptor;
  bool enableParallel = false;
  bool disableSaturation = false;

  ONNXElementwiseOpLoweringWithNNPALayout(TypeConverter &typeConverter,
      MLIRContext *ctx, bool enableParallel, bool disableSaturation)
      : OpConversionPattern<ElementwiseOp>(typeConverter, ctx,
            PatternBenefit(
                10)), // Benefit must be high so that we come here first.
        disableSaturation(disableSaturation) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ElementwiseOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    Location loc = ONNXLoc<ElementwiseOp>(op);
    ValueRange operands = adaptor.getOperands();
    int64_t numArgs = operands.size();

    LLVM_DEBUG(llvm::dbgs() << "Investigate elementwise op (" << op->getName()
                            << ") with possible NNPA layout\n");

    // Test if operation is suitable for processing here. If not, will be
    // handled by the normal elementwise operations.
    if (!isZTensorOfF32AndDLF16(op)) {
      return failure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Process elementwise op " << op->getName()
                   << " with NNPA layout:\n  ";
      op->dump();
    });
    assert(numArgs >= 0 && numArgs <= 2 && op->getNumResults() == 1 &&
           "expect at most 2 inputs, exactly 1 output");

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Value alloc;
    Value outputTensor = elmsOp.getResult();
    Type outputTensorType = outputTensor.getType();
    if (zhigh::isZTensor(outputTensorType)) {
      // Alloc for Z MemRefs
      zhigh::ZMemRefType zMemRefType =
          zhigh::convertZTensorToMemRefType(outputTensorType);
      // Allocate a buffer for the result MemRef.
      alloc = zhigh::insertAllocForZMemRef(
          zMemRefType, shapeHelper.getOutputDims(), op, rewriter);
    } else {
      // Convert the output type to MemRefType.
      Type convertedType = this->typeConverter->convertType(outputTensorType);
      int64_t alignment =
          KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
      assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
             "Failed to convert type to MemRefType");
      MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
      // Insert an allocation and deallocation for the result of this operation.
      alloc = allocOrReuse(create.mem, op, operands, outputMemRefType,
          shapeHelper.getOutputDims(), alignment);
    }

    MultiValuesOfF32IterateBodyFn fct =
        [&](const KrnlBuilder &b,
            mlir::SmallVectorImpl<mlir::Value> &inputOfF32Vals) {
          return emitScalarOpFor<ElementwiseOp>(rewriter, b.getLoc(), op,
              inputOfF32Vals[0].getType(), inputOfF32Vals);
        };
    // Unroll: can unroll up to 8 (for 8 * simd of 8 = 1 stick of 64.)
    int64_t unrollFactor = 8;
    IterateOverStickInputOutput(create.krnl, op, adaptor.getOperands(), alloc,
        shapeHelper.getOutputDims(), unrollFactor, enableParallel,
        disableSaturation, true /*prefetch*/, fct);

    // replace op.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generate code for (well behaving) Layer Norm, namely axis = -1, inner dim of
// static size and multiple of 64.

template <typename OP_TYPE>
void generateLayerNormOptimizedCode(MDBuilder &create, OP_TYPE lnOp,
    /* inputs */ Value XMemRef, Value scaleMemRef, Value biasMemRef,
    /* outputs */ Value YMemRef,
    /* temps [B][vec] */ Value redMemRef, Value redMemRef2,
    /* index expr param */ IndexExpr redDim,
    /* value params */ Value i, Value epsilon,
    /* int params */ int64_t B, int64_t totVL,
    IndexExpr scaleModFactor, IndexExpr biasModFactor) {}

//===----------------------------------------------------------------------===//
// Pass

namespace zhigh {

void populateONNXWithNNPALayoutToKrnlConversionPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx,
    bool enableParallel, bool disableSaturation) {
// Add the insert patterns for all elementwise types regardless of unary, binary
// or variadic.
#define ELEMENTWISE_ALL(_OP_TYPE)                                              \
  patterns.insert<ONNXElementwiseOpLoweringWithNNPALayout<_OP_TYPE>>(          \
      typeConverter, ctx, enableParallel, disableSaturation);
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.hpp"
}

} // namespace zhigh
} // namespace onnx_mlir
