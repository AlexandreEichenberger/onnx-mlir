// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --nnpa-zhigh-inplace-elementwise --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s --check-prefix=OFF

// Tests for --nnpa-zhigh-inplace-elementwise: zhigh.Add and zhigh.Mul write
// their result into an input buffer that dies at the op.

// The Add writes into the MatMul buffer: no alloc for the result.
func.func @add_inplace_into_matmul(%arg0: tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, %arg1: tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, %arg2: tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>, %arg3: tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  %1 = "zhigh.Add"(%0, %arg3) : (tensor<*xf16>, tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<*xf16>
  return %1 : tensor<*xf16>
// CHECK-LABEL:  func.func @add_inplace_into_matmul
// CHECK-SAME:   ({{.*}}, [[PARAM_3_:%.+]]: memref<4x16xf16, #map>) -> memref<4x16xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf16, #map>
// CHECK:           "zlow.matmul"({{.*}}, [[RES_]])
// CHECK-NOT:       memref.alloc() {{.*}}: memref<4x16xf16, #map>
// CHECK:           "zlow.add"([[RES_]], [[PARAM_3_]], {{.*}}, [[RES_]]) <{layout = "2D"}>
// CHECK:           return [[RES_]] : memref<4x16xf16, #map>

// OFF-LABEL:    func.func @add_inplace_into_matmul
// OFF-SAME:     ({{.*}}, [[PARAM_3_:%.+]]: memref<4x16xf16, #map>) -> memref<4x16xf16, #map> {
// OFF:             [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf16, #map>
// OFF:             "zlow.matmul"({{.*}}, [[RES_]])
// OFF:             [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf16, #map>
// OFF:             "zlow.add"([[RES_]], [[PARAM_3_]], {{.*}}, [[RES_1_]]) <{layout = "2D"}>
// OFF:             return [[RES_1_]] : memref<4x16xf16, #map>
}

// -----

// Chain Add(Add(mm1, arg), mm2): both Adds write into the buffer of mm1.
func.func @add_inplace_chain(%arg0: tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, %arg1: tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, %arg2: tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>, %arg3: tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  %1 = "zhigh.Add"(%0, %arg3) : (tensor<*xf16>, tensor<4x16xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<*xf16>
  %2 = "zhigh.MatMul"(%arg0, %arg1, %arg2) : (tensor<4x8xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<8x16xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
  %3 = "zhigh.Add"(%1, %2) : (tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
  return %3 : tensor<*xf16>
// CHECK-LABEL:  func.func @add_inplace_chain
// CHECK-SAME:   ({{.*}}, [[PARAM_3_:%.+]]: memref<4x16xf16, #map>) -> memref<4x16xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf16, #map>
// CHECK:           "zlow.matmul"({{.*}}, [[RES_]])
// CHECK-NOT:       memref.alloc() {{.*}}: memref<4x16xf16, #map>
// CHECK:           "zlow.add"([[RES_]], [[PARAM_3_]], {{.*}}, [[RES_]]) <{layout = "2D"}>
// CHECK:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf16, #map>
// CHECK:           "zlow.matmul"({{.*}}, [[RES_2_]])
// CHECK-NOT:       memref.alloc() {{.*}}: memref<4x16xf16, #map>
// CHECK:           "zlow.add"([[RES_]], [[RES_2_]], {{.*}}, [[RES_]]) <{layout = "2D"}>
// CHECK:           return [[RES_]] : memref<4x16xf16, #map>
}

// -----

// Mul(arg, Sigmoid(arg)): operand 0 is a function argument, so the Mul writes
// into operand 1, the Sigmoid buffer.
func.func @mul_inplace_into_operand1(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Sigmoid"(%arg1) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  %1 = "zhigh.Mul"(%arg0, %0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<*xf16>) -> tensor<*xf16>
  return %1 : tensor<*xf16>
// CHECK-LABEL:  func.func @mul_inplace_into_operand1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>, [[PARAM_1_:%.+]]: memref<3x4x5xf16, #map>) -> memref<3x4x5xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.sigmoid"([[PARAM_1_]], {{.*}}, [[RES_]])
// CHECK-NOT:       memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.mul"([[PARAM_0_]], [[RES_]], {{.*}}, [[RES_]]) <{layout = "3D"}>
// CHECK:           return [[RES_]] : memref<3x4x5xf16, #map>
}

// -----

// Negative: the Sigmoid result has a second use (it is also returned).
func.func @no_inplace_two_uses(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<*xf16>, tensor<*xf16>) {
  %0 = "zhigh.Sigmoid"(%arg1) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  %1 = "zhigh.Mul"(%arg0, %0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<*xf16>) -> tensor<*xf16>
  return %0, %1 : tensor<*xf16>, tensor<*xf16>
// CHECK-LABEL:  func.func @no_inplace_two_uses
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>, [[PARAM_1_:%.+]]: memref<3x4x5xf16, #map>)
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.sigmoid"([[PARAM_1_]], {{.*}}, [[RES_]])
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.mul"([[PARAM_0_]], [[RES_]], {{.*}}, [[RES_1_]]) <{layout = "3D"}>
// CHECK:           return [[RES_]], [[RES_1_]] : memref<3x4x5xf16, #map>, memref<3x4x5xf16, #map>
}

// -----

// Negative: Add(X, X) uses X twice.
func.func @no_inplace_same_operand(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Sigmoid"(%arg0) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  %1 = "zhigh.Add"(%0, %0) : (tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
  return %1 : tensor<*xf16>
// CHECK-LABEL:  func.func @no_inplace_same_operand
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.sigmoid"({{.*}}, [[RES_]])
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.add"([[RES_]], [[RES_]], {{.*}}, [[RES_1_]]) <{layout = "3D"}>
// CHECK:           return [[RES_1_]] : memref<3x4x5xf16, #map>
}

// -----

// Negative: both operands are function arguments.
func.func @no_inplace_func_args(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Add"(%arg0, %arg1) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>
// CHECK-LABEL:  func.func @no_inplace_func_args
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>, [[PARAM_1_:%.+]]: memref<3x4x5xf16, #map>) -> memref<3x4x5xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.add"([[PARAM_0_]], [[PARAM_1_]], {{.*}}, [[RES_]]) <{layout = "3D"}>
// CHECK:           return [[RES_]] : memref<3x4x5xf16, #map>
}

// -----

// Negative: zhigh.Reshape may lower to a view of its input, so its result is
// not reused.
func.func @no_inplace_reshape(%arg0: tensor<3x4x50xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<3x20x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %shape = onnx.Constant dense<[3, 20, 10]> : tensor<3xi64>
  %0 = "zhigh.Reshape"(%arg0, %shape) {layout = "3DS"} : (tensor<3x4x50xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<3xi64>) -> tensor<*xf16>
  %1 = "zhigh.Add"(%0, %arg1) : (tensor<*xf16>, tensor<3x20x10xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  return %1 : tensor<*xf16>
// CHECK-LABEL:  func.func @no_inplace_reshape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x50xf16, #map>, [[PARAM_1_:%.+]]: memref<3x20x10xf16, #map>) -> memref<3x20x10xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x20x10xf16, #map>
// CHECK:           "zlow.reshape"([[PARAM_0_]], [[RES_]])
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x20x10xf16, #map>
// CHECK:           "zlow.add"([[RES_]], [[PARAM_1_]], {{.*}}, [[RES_1_]]) <{layout = "3DS"}>
// CHECK:           return [[RES_1_]] : memref<3x20x10xf16, #map>
}

// -----

// Negative: only Add and Mul are written in place; Sub gets its own buffer.
func.func @no_inplace_sub(%arg0: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Sigmoid"(%arg1) : (tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  %1 = "zhigh.Sub"(%0, %arg0) : (tensor<*xf16>, tensor<3x4x5xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<*xf16>
  return %1 : tensor<*xf16>
// CHECK-LABEL:  func.func @no_inplace_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf16, #map>, [[PARAM_1_:%.+]]: memref<3x4x5xf16, #map>) -> memref<3x4x5xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.sigmoid"([[PARAM_1_]], {{.*}}, [[RES_]])
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf16, #map>
// CHECK:           "zlow.sub"([[RES_]], [[PARAM_0_]], {{.*}}, [[RES_1_]]) <{layout = "3D"}>
// CHECK:           return [[RES_1_]] : memref<3x4x5xf16, #map>
}

// -----

// Dynamic dims: the Mul writes into the Sigmoid buffer.
func.func @mul_inplace_dynamic(%arg0: tensor<?x?x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<?x?x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16> {
  %0 = "zhigh.Sigmoid"(%arg0) : (tensor<?x?x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  %1 = "zhigh.Mul"(%0, %arg1) : (tensor<*xf16>, tensor<?x?x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<*xf16>
  return %1 : tensor<*xf16>
// CHECK-LABEL:  func.func @mul_inplace_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x384xf16, #map>, [[PARAM_1_:%.+]]: memref<?x?x384xf16, #map>) -> memref<?x?x384xf16, #map> {
// CHECK:           [[RES_:%.+]] = memref.alloc({{.*}}) {{.*}}: memref<?x?x384xf16, #map>
// CHECK:           "zlow.sigmoid"([[PARAM_0_]], {{.*}}, [[RES_]])
// CHECK-NOT:       memref.alloc({{.*}}) {{.*}}: memref<?x?x384xf16, #map>
// CHECK:           "zlow.mul"([[RES_]], [[PARAM_1_]], {{.*}}, [[RES_]]) <{layout = "3DS"}>
// CHECK:           return [[RES_]] : memref<?x?x384xf16, #map>
}
