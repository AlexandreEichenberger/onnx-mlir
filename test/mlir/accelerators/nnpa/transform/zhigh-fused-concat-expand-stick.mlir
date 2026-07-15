// RUN: onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file | FileCheck %s

// Tests for the zhigh.concat-expand-stick FusedOp pattern.
// The pass wraps the chain:
//   ONNXConcatOp -> ONNXUnsqueezeOp -> ZHighF32ToDLF16Op -> ONNXExpandOp ->
//   ONNXReshapeOp -> ONNXLayoutTransformOp
// into an onnx.Fused region with kind = "zhigh.concat-expand-stick". This is
// the GQA/MQA "repeat KV heads after cache-concat" idiom: concatenate
// past/present KV cache on the sequence axis, then broadcast (repeat) the
// key-value-head dimension up to the query-head count before re-stickifying.
//
// Pattern under test:
//   Inputs: tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>
//   Concat axis 2         -> tensor<2x4x8x64xf32>   (A=2; both inputs' 64 is
//                                                     mod-64)
//   Unsqueeze axis 2      -> tensor<2x4x1x8x64xf32>  (P=2)
//   F32ToDLF16            -> tensor<2x4x1x8x64xf16>
//   Expand dim P to N=3   -> tensor<2x4x3x8x64xf16>
//   Reshape [24, 8, 64]   -> tensor<24x8x64xf16>     (dims 0..2 collapsed:
//                                                     2*4*3=24)
//   LayoutTransform 3DS   -> tensor<24x8x64xf16, 3DS>
//
// Expected stored params:
//   concatAxis = 2, unsqueezedPosition = 2, expansionN = 3,
//   noSaturation = false, reshapeFirstCollapsedDim = 0,
//   reshapeCollapsedCount = 3, finalLayout = "3DS"

// -----

func.func @concat_expand_stick_basic(
    %arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>)
    -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @concat_expand_stick_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x3x64xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5x64xf32>)
// Fused op is created with exactly two external inputs (the function
// arguments); the constants are cloned inside the body.
// CHECK:           [[VAR_0_:%.+]] = "onnx.Fused"([[PARAM_0_]], [[PARAM_1_]]) <{kind = "zhigh.concat-expand-stick"}>
// CHECK-DAG:           onnx.Constant dense<2>
// CHECK-DAG:           onnx.Constant{{.*}}[2, 4, 3, 8, 64]
// CHECK-DAG:           onnx.Constant{{.*}}[24, 8, 64]
// Verify the six chain ops are inside the fused body:
// CHECK:           "onnx.Concat"{{.*}}-> tensor<2x4x8x64xf32>
// CHECK:           "onnx.Unsqueeze"{{.*}}-> tensor<2x4x1x8x64xf32>
// CHECK:           "zhigh.F32ToDLF16"{{.*}}-> tensor<2x4x1x8x64xf16>
// CHECK:           "onnx.Expand"{{.*}}-> tensor<2x4x3x8x64xf16>
// CHECK:           "onnx.Reshape"{{.*}}-> tensor<24x8x64xf16>
// CHECK:           "onnx.LayoutTransform"{{.*}}-> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           onnx.Yield
// Verify stored params (attrs printed alphabetically after the body closes):
// CHECK:           concatAxis = 2{{.*}}expansionN = 3{{.*}}finalLayout = "3DS"{{.*}}noSaturation = false{{.*}}reshapeCollapsedCount = 3{{.*}}reshapeFirstCollapsedDim = 0{{.*}}unsqueezedPosition = 2
// CHECK:           return [[VAR_0_]] : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           }
}

// -----

// Concat: rejected when there are more than two inputs.

func.func @no_fuse_concat_three_inputs(%arg0: tensor<2x4x3x64xf32>,
    %arg1: tensor<2x4x2x64xf32>, %arg2: tensor<2x4x3x64xf32>)
    -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1, %arg2) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x2x64xf32>, tensor<2x4x3x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @no_fuse_concat_three_inputs
// detectIfBeneficial rejects the chain: Concat has 3 inputs, not 2.
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Concat"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "zhigh.F32ToDLF16"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Reshape"
// CHECK:           "onnx.LayoutTransform"
// CHECK:           return
}

// -----

// Concat: rejected when an input's innermost dim is not static mod 64.

func.func @no_fuse_concat_innermost_not_mod64(
    %arg0: tensor<2x4x3x50xf32>, %arg1: tensor<2x4x5x50xf32>)
    -> tensor<24x8x50xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 50]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[24, 8, 50]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x50xf32>, tensor<2x4x5x50xf32>) -> tensor<2x4x8x50xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x50xf32>, tensor<1xi64>) -> tensor<2x4x1x8x50xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<2x4x1x8x50xf32>) -> tensor<2x4x1x8x50xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x4x1x8x50xf16>, tensor<5xi64>) -> tensor<2x4x3x8x50xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x50xf16>, tensor<3xi64>) -> tensor<24x8x50xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<24x8x50xf16>) -> tensor<24x8x50xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x50xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @no_fuse_concat_innermost_not_mod64
// detectIfBeneficial rejects the chain: innermost dim 50 is not mod 64.
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Concat"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "zhigh.F32ToDLF16"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Reshape"
// CHECK:           "onnx.LayoutTransform"
// CHECK:           return
}

// -----

// detectUpperCollapse — tail dim changed by reshape (mirrors
// no_fuse_tail_dim_changed in zhigh-fused-expand-mul-stick.mlir).
// P=1; dims strictly after P (indices 2,3: sizes 4,64) must stay unchanged,
// but the reshape merges them into a single dim of size 256.

func.func @no_fuse_reshape_tail_dim_changed(
    %arg0: tensor<3x2x64xf32>, %arg1: tensor<3x2x64xf32>)
    -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
  %shre  = onnx.Constant dense<[24, 256]>     : tensor<2xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 1 : si64}>
            : (tensor<3x2x64xf32>, tensor<3x2x64xf32>) -> tensor<3x4x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<3x1x4x64xf32>) -> tensor<3x1x4x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<3x1x4x64xf16>, tensor<4xi64>) -> tensor<3x8x4x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<3x8x4x64xf16>, tensor<2xi64>) -> tensor<24x256xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "2DS"}>}
            : (tensor<24x256xf16>) -> tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>
  return %out : tensor<24x256xf16, #zhigh.layout<{dataLayout = "2DS"}>>

// CHECK-LABEL:  func.func @no_fuse_reshape_tail_dim_changed
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Concat"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "zhigh.F32ToDLF16"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Reshape"
// CHECK:           "onnx.LayoutTransform"
// CHECK:           return
}

// -----

// Expand: rejected when a non-P dim changes (mirrors
// no_fuse_expand_changes_non_p_dim in zhigh-fused-expand-mul-stick.mlir).
// P=1; dim 0 changes from 3 to 6, which is not the unsqueezed dim.

func.func @no_fuse_expand_changes_non_p_dim(
    %arg0: tensor<3x2x64xf32>, %arg1: tensor<3x2x64xf32>)
    -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<1>             : tensor<1xi64>
  %shexp = onnx.Constant dense<[6, 8, 4, 64]> : tensor<4xi64>
  %shre  = onnx.Constant dense<[48, 4, 64]>   : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 1 : si64}>
            : (tensor<3x2x64xf32>, tensor<3x2x64xf32>) -> tensor<3x4x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<3x1x4x64xf32>) -> tensor<3x1x4x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<3x1x4x64xf16>, tensor<4xi64>) -> tensor<6x8x4x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<6x8x4x64xf16>, tensor<3xi64>) -> tensor<48x4x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<48x4x64xf16>) -> tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<48x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL:  func.func @no_fuse_expand_changes_non_p_dim
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Concat"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "zhigh.F32ToDLF16"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Reshape"
// CHECK:           "onnx.LayoutTransform"
// CHECK:           return
}

// -----

// LayoutTransform: rejected when the target layout is not one of
// {3D, 3DS, 4D} (mirrors no_fuse_stick_unsupported_layout in
// zhigh-fused-expand-mul-stick.mlir). "2D" is otherwise a supported
// compiler-generated-stick layout, so this specifically exercises the
// {3D, 3DS, 4D} allow-list check rather than
// supportedLayoutForCompilerGeneratedStickUnstick.

func.func @no_fuse_layout_transform_unsupported_layout(
    %arg0: tensor<32x64xf32>, %arg1: tensor<32x64xf32>)
    -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>> {
  %axes  = onnx.Constant dense<0>              : tensor<1xi64>
  %shexp = onnx.Constant dense<[8, 64, 64]>    : tensor<3xi64>
  %shre  = onnx.Constant dense<[512, 64]>      : tensor<2xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 0 : si64}>
            : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<64x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<64x64xf32>, tensor<1xi64>) -> tensor<1x64x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<1x64x64xf32>) -> tensor<1x64x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<1x64x64xf16>, tensor<3xi64>) -> tensor<8x64x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<8x64x64xf16>, tensor<2xi64>) -> tensor<512x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "2D"}>}
            : (tensor<512x64xf16>) -> tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  return %out : tensor<512x64xf16, #zhigh.layout<{dataLayout = "2D"}>>

// CHECK-LABEL:  func.func @no_fuse_layout_transform_unsupported_layout
// CHECK-NOT:       "onnx.Fused"
// CHECK:           "onnx.Concat"
// CHECK:           "onnx.Unsqueeze"
// CHECK:           "zhigh.F32ToDLF16"
// CHECK:           "onnx.Expand"
// CHECK:           "onnx.Reshape"
// CHECK:           "onnx.LayoutTransform"
// CHECK:           return
}
