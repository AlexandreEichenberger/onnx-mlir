/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnnxMlirCompiler.h"
#include "CompilerUtils.hpp"

extern "C" {
namespace onnx_mlir {

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromEnv(const char *envVarName) {
  // ParseCommandLineOptions needs at least one argument
  std::string nameStr = "program-name";
  const char *argv[1];
  argv[0] = nameStr.c_str();
  const char *name = envVarName ? envVarName : OnnxMlirEnvOptionName.c_str();
  return llvm::cl::ParseCommandLineOptions(
      1, argv, "SetCompilerOptionsFromEnv\n", nullptr, name);
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgs(
    int64_t argc, char *argv[]) {
  return llvm::cl::ParseCommandLineOptions(
      argc, argv, "SetCompilerOptionsFromEnv\n", nullptr, nullptr);
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromEnvAndArgs(
    const char *envVarName, int64_t argc, char *argv[]) {
  const char *name = envVarName ? envVarName : OnnxMlirEnvOptionName.c_str();
  return llvm::cl::ParseCommandLineOptions(
      argc, argv, "SetCompilerOptionsFromEnv\n", nullptr, name);
}

ONNX_MLIR_EXPORT int64_t omSetCompilerOptions(
    const OptionKind kind, const char *val) {
  return setCompilerOption(kind, std::string(val));
}

ONNX_MLIR_EXPORT const char *omGetCompilerOption(const OptionKind kind) {
  std::string val = getCompilerOption(kind);
  return val.c_str();
}

ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char **errorMessage) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;
  registerDialects(context);

  std::string error_message;
  processInputFile(std::string(inputFilename), context, module, &error_message);
  if (errorMessage != NULL) {
    *errorMessage = error_message.c_str();
    return 1;
  }
  return compileModule(module, context, outputBaseName, emissionTarget);
}

ONNX_MLIR_EXPORT int64_t omCompileFromArray(const void *inputBuffer,
    int bufferSize, const char *outputBaseName,
    EmissionTargetType emissionTarget) {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;
  registerDialects(context);

  processInputArray(inputBuffer, bufferSize, context, module);
  return compileModule(module, context, outputBaseName, emissionTarget);
}

} // namespace onnx_mlir
}
