# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_executable(TestSmallFP
  TestSmallFP.cpp

  NO_INSTALL

  LINK_LIBS PRIVATE
  OMMlirUtilities
  benchmark
  )

add_test(NAME TestSmallFP COMMAND TestSmallFP)
