/*
 * SPDX-License-Identifier: Apache-2.0
 */

//==============-- ConvModel.cpp - Building Conv Models for tests -===========//
//
// Copyright 2022-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a function that build a convolution model and compiles it.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"
#include "test/modellib/ModelLib.hpp"

#define DEBUG 0
#define HI_ALEX 1

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

const string getAutoPadName(const int autoPad) {
  static const string autoPadName[] = {
      "NOTSET", "VALID", "SAME_LOWER", "SAME_UPPER"};
  assert(autoPad >= 0 && autoPad < AUTO_PAD_UB && "out of bound autopad");
  return autoPadName[autoPad];
}

#if HI_ALEX
// Local support.
static int myCeil(int a, int b) { return ceil((1.0 * a) / (1.0 * b)); }
static int myFloor(int a, int b) { return floor((1.0 * a) / (1.0 * b)); }

//===----------------------------------------------------------------------===//
// Compute Shape for various auto pad value.
//===----------------------------------------------------------------------===//

// TODO: Ideally these values would be corroborated with Pytorch/TF. However,
// Pytorch only supports same/valid with unit strides. Maybe check with TF?
static LogicalResult checkShapes(
    /*in*/
    const int NIn, const int CIn, const int HIn, const int WIn, const int kH,
    const int kW, const int autoPad, const int stride, const int dilation,
    const int NOut, const int COut, const int HOut, const int WOut,
    /*in/out*/
    int &pHBegin, int &pHEnd, int &pWBegin, int &pWEnd) {

  // Check first params.
  if (NIn != NOut) {
    cerr << "N mismatch: in " << NIn << ", out " << NOut << endl;
    return failure();
  }
  if (CIn != COut) {
    cerr << "C mismatch: in " << CIn << ", out " << COut << endl;
    return failure();
  }

  // Gather variables in arrays to match ONNX descriptions.
  int I[] = {HIn, WIn};
  int K[] = {kH, kW};
  int pBegin[] = {pHBegin, pWBegin};
  int pEnd[] = {pHEnd, pWEnd};
  int p[] = {pHBegin + pHEnd, pWBegin + pWEnd};
  int s[] = {stride, stride};
  int d[] = {dilation, dilation};
  int O[] = {HOut, WOut};

  // Check dimensions for the spatial axes. From MaxPool:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
  int myO[2], myPBegin[2], myPEnd[2];
  for (int i = 0; i < 2; ++i) {
    if (autoPad == AUTO_PAD_NOTSET) {
      // NOSET:
      //  * O[i] = floor((I[i] + P[i] - ((K[i] - 1) * d[i] + 1)) / s[i] + 1)
      myO[i] = myFloor((I[i] + p[i] - ((K[i] - 1) * d[i] + 1)), s[i]) + 1;
      myPBegin[i] = pBegin[i];
      myPEnd[i] = pEnd[i];
    } else if (autoPad == AUTO_PAD_VALID) {
      // VALID:
      // * O[i] = ceil((I[i] - ((K[i] - 1) * d[i] + 1) + 1) / s[i])
      // * P = 0
      myO[i] = myCeil((I[i] - ((K[i] - 1) * d[i] + 1) + 1), s[i]);
      myPBegin[i] = myPEnd[i] = 0;
    } else {
      // SAME_LOWER or SAME_UPPER:
      // * O[i] = ceil(I[i] / s[i])
      // * p' = (O[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i]
      // * P[i] = p' / 2, if odd, first or second are increased by one.
      myO[i] = myCeil(I[i], s[i]);
      int pSum = (myO[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i];
      pSum = pSum >= 0 ? pSum : 0;
      myPBegin[i] = myPEnd[i] = pSum / 2;
      if (pSum % 2 != 0) {
        if (autoPad == AUTO_PAD_UPPER)
          myPEnd[i] += 1;
        else
          myPBegin[i] += 1;
      }
    }
    if (myO[i] != O[i]) {
      cerr << "output sizes mismatch: computed " << myO[i] << ", got " << O[i]
           << endl;
      return failure();
    }
  }
  // Test all good, set padding values for computed ones.
  pHBegin = myPBegin[0];
  pWBegin = myPBegin[1];
  pHEnd = myPEnd[0];
  pWEnd = myPEnd[1];

  return success();
}
#endif

//===----------------------------------------------------------------------===//
// Evaluate Convolution
//===----------------------------------------------------------------------===//

bool generateCompiledConv2DModel(const string modelName,
    /*in*/
    const int N, const int C, const int H, const int W, const int kH,
    const int kW, const int autoPad, const int stride, const int dilation,
    const int isDynamic,
    /* in/out */
    int &pHBegin, int &pHEnd, int &pWBegin, int &pWEnd,
    /* out */
    int &NOut, int &COut, int &HOut, int &WOut) {

  if (autoPad != AUTO_PAD_NOTSET) {
    // make sure all pads are initially zero, only value tolarated.
    assert(pHBegin == 0 && pHEnd == 0 && pWBegin == 0 && pWEnd == 0);
  }

  MLIRContext ctx;
  setCompileContext(ctx, {{OptionKind::CompilerOptLevel, "3"}});

  // We use the Ns for the shape of the input, and the N1s for the construction
  // of the model. That way, when the shape is dynamic, we set the N1s to "-1"
  // (dynamic value) so that the compiler may not infer the size of the model,
  // and instead generate code to figure the sizes at run time.
  int N1 = N;
  int C1 = C;
  int H1 = H;
  int W1 = W;
  if (isDynamic)
    N1 = C1 = H1 = W1 = -1;

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 4> xShape = {N, C, H, W};
  llvm::SmallVector<int64_t, 3> xShapeSymbol = {N1, C1, H1, W1};
  llvm::SmallVector<int64_t, 1> bShape = {C};
  llvm::SmallVector<int64_t, 4> wShape = {C, C, kH, kW};
  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto xTypeSymbol = RankedTensorType::get(xShapeSymbol, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{xTypeSymbol, wType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "main_graph";
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<FuncOp>(UnknownLoc::get(&ctx), funcName, funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto xVal = entryBlock->getArgument(0);
  auto wVal = entryBlock->getArgument(1);
  auto bVal =
      builder.create<ConstantOp>(UnknownLoc::get(&ctx), builder.getUnitAttr())
          .getResult();

  auto dilations = builder.getI64ArrayAttr({dilation, dilation});
  auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
  auto pads = builder.getI64ArrayAttr({pHBegin, pWBegin, pHEnd, pWEnd});
  auto strides = builder.getI64ArrayAttr({stride, stride});

  auto convOp = builder.create<ONNXConvOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType,
      /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
      /*auto_pad=*/builder.getStringAttr(getAutoPadName(autoPad)),
      /*dilations=*/dilations,
      /*group=*/
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, 1, /*isSigned=*/true)),
      /*kernel_shape=*/kernel_shape, /*pads=*/pads,
      /*strides=*/strides);

  // Use the convOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in a inconsistent state.
  convOp.X().setType(xType); // Use static dims to infer shape.
  LogicalResult res = convOp.inferShapes([](mlir::Region &) {});
  if (failed(res)) {
    return false;
  }
  auto outputShape = convOp.getResult().getType().cast<ShapedType>().getShape();
  NOut = outputShape[0];
  COut = outputShape[1];
  HOut = outputShape[2];
  WOut = outputShape[3];
  convOp.getResult().setType(yType);
  convOp.X().setType(xTypeSymbol);
  #if HI_ALEX
  res = checkShapes(N, C, H, W, kH, kW, autoPad, stride, dilation, NOut, COut,
      HOut, WOut, pHBegin, pHEnd, pWBegin, pWEnd);
  if (failed(res)) {
    if (DEBUG) {
      cerr << "Conv after check shape, N out " << NOut << ", C out " << COut
           << ", H out " << HOut << ", W out " << WOut << ", ph begin "
           << pHBegin << ", ph end " << pHEnd << ", pw begin " << pWBegin
           << ", pw end " << pWEnd << endl;
    }
    return false;
  }
  #endif

  llvm::SmallVector<Value, 1> results = {convOp.getResult()};
  builder.create<ReturnOp>(UnknownLoc::get(&ctx), results);
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  std::string signature("");
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/2,
      /*numOutputs=*/1,
      /*signature*/ signature);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);
  compileModule(moduleRef, ctx, modelName, onnx_mlir::EmitLib);

  return true;
}
