//===-- VectorMachineSupport.hpp - Helper for what SIMD ops are supported -===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Support to determine which high level op is supported on a given target.
// Does not need to be exact (as LLVM backend can lower any vector code),
// however it is at time useful to have a rough idea of what is eventually
// supported by the target hardware to better direct high level optimizations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Generic ops to determine if operations are supported for lowering to vector
// SIMD operations. An additional operation's element type will be provided to
// further refine whether an operation can be effectively vectorized on the
// given target hardware. This list roughly corresponds to the operations
// supported by MathDialectBuilder, with some combining of similar operations
// (e.g. all the compares).

enum class GenericOps {
  AbsGOp,
  ArithmeticGOp, /* Simple compute ops: add/sub */
  CeilDivGOp,
  CeilGOp,
  CompareGOp, /* All compare operations, signed/unsigned fixed/float. */
  ConversionGOp,
  CopySignGOP,
  DivGOp,
  Exp2GOp,
  ExpGOp,
  FloorDivGOp,
  FloorGOp,
  FmaGOp,
  Log2GOp,
  LogGOp,
  LogicalGOp, /* All logical ops: and, or, xor, not, nor, nand,... */
  MinMaxGOp,
  MulGOp,
  PowGOp,
  RemGOp,
  SelectGOp,
  ShiftGOp,   /* Shift operations: logical/arithmetic. */
  ShuffleGOp, /* All bit/byte moving operations: shuffle, rotate, shift. */
  SqrtGOp,
  SumAcrossGOp, /* Sum across vector. */
};

//===----------------------------------------------------------------------===//
// Generic vector machine support class, which must be refined for each
// supported machine type.

class VectorMachineSupport {
protected:
  VectorMachineSupport() = default;
  virtual ~VectorMachineSupport() = default;

public:
  // The class encapsulate a single static vector machine support
  VectorMachineSupport *getGlobalVectorMachineSupport();
  void setGlobalVectorMachineSupport(std::string name);
  void clearGlobalVectorMachineSupport();

  static const int64_t UNSUPPORTED = 0;

  // Return the bit width of the SIMD unit regardless of the type/operation.
  virtual int64_t getVectorBitWidth() = 0;
  // Return the number of elements that can be processed in SIMD fashion
  // regardless of the type/operation.
  virtual int64_t getVectorLength(mlir::Type elementType);
  // Return the number of elements that can be processed in SIMD fashion if
  // support exists; 0 when there is no support.
  virtual int64_t getVectorLength(GenericOps gop, mlir::Type elementType) = 0;

  // Composite function that return the average vector length among the
  // operations that support SIMD execution.
  double getAvgVectorLength(llvm::SmallVectorImpl<GenericOps> &gops,
      mlir::Type elementType, int64_t &numSupported, int64_t &numUnsupported);

private:
  static VectorMachineSupport *globalVectorMachineSupport;
};

// No support for SIMD.
class NoVectorMachineSupport : public VectorMachineSupport {
public:
  NoVectorMachineSupport() = default;
  virtual ~NoVectorMachineSupport() = default;

  int64_t getVectorBitWidth() override { return 0; }
  int64_t getVectorLength(mlir::Type elementType) override {
    return UNSUPPORTED;
  }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override {
    return UNSUPPORTED;
  }
};

// Support for IBM Z servers.

class Z16VectorMachineSupport : public VectorMachineSupport {
public:
  Z16VectorMachineSupport() = default;
  virtual ~Z16VectorMachineSupport() = default;

  int64_t getVectorBitWidth() override { return 128; }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override;
};

// TODO: create models for z14 and z15.
using Z14VectorMachineSupport = Z16VectorMachineSupport;
using Z15VectorMachineSupport = Z16VectorMachineSupport;

// Support for x86 processors (SSE 4.2 and AVX2)
class SSE42x86VectorMachineSupport : public VectorMachineSupport {
public:
  SSE42x86VectorMachineSupport() = default;
  virtual ~SSE42x86VectorMachineSupport() = default;

  int64_t getVectorBitWidth() override { return 128; }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override;
};

class AVX2x86VectorMachineSupport : public SSE42x86VectorMachineSupport {
public:
  AVX2x86VectorMachineSupport() = default;
  virtual ~AVX2x86VectorMachineSupport() = default;

  int64_t getVectorBitWidth() override { return 258; }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override;
};

} // namespace onnx_mlir