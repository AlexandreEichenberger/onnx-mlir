/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- runmodel.cpp  ------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
/*
  This file help run a onnx model as simply as possible for testing.
  Compile as follows in the onnx-mlir build subdirectory. The tool is built as
  follows. For dinamically loaded models:

cd onnx-mlir/build
. ../utils/build-run-onnx-lib.sh
run-onnx-lib test/backend/test_add.so

  For statically loaded models, best is to run the utility in the directory
  of the model.

cd onnx-mlir/build
. ../utils/build-run-onnx-lib.sh test/backend/test_add.so
cd test/backend
run-onnx-lib

  Usage of program is as follows.

Usage: run-onnx-lib [options] model.so

  Program will instantiate the model given by "model.so"
  with random inputs, launch the computation, and ignore
  the results. A model is typically generated by lowering
  an ONNX model using a "onnx-mlir --EmitLib model.onnx"
  command. When the input model is not found as is, the
  path to the local directory is also prepended.

  Options:
    -e name | --entry-point name
         Name of the ONNX model entry point.
         Default is "run_main_graph".
    -n NUM | --iterations NUM
         Number of times to run the tests, default 1
    -v | --verbose
         Print the shape of the inputs and outputs
    -h | --help
         help
*/

//===----------------------------------------------------------------------===//

// Define while compiling.
// #define LOAD_MODEL_STATICALLY 1

#include <algorithm>
#include <assert.h>
#include <dlfcn.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

// Json reader & LLVM suport.
#include "OnnxMlirRuntime.h"
#include "llvm/Support/JSON.h"

using namespace std;

#ifdef WIN32
// TO BE FIXED
#warning disabled
#else
#include <sys/time.h>
// Util for timing.
struct timeval startTime, stopTime, result;
#endif
// Data structure to hold measurement times (in microseconds).
vector<uint64_t> timeLogInMicroSec;

// Interface definitions
extern "C" OMTensorList *run_main_graph(OMTensorList *);
extern "C" const char *omInputSignature();
extern "C" const char *omOutputSignature();
extern "C" OMTensor *omTensorCreate(void *, int64_t *, int64_t, OM_DATA_TYPE);
extern "C" OMTensorList *TensorListCreate(OMTensor **, int);
extern "C" void omTensorListDestroy(OMTensorList *list);
// DLL definitions
OMTensorList *(*dll_run_main_graph)(OMTensorList *);
const char *(*dll_omInputSignature)();
const char *(*dll_omOutputSignature)();
OMTensor *(*dll_omTensorCreate)(void *, int64_t *, int64_t, OM_DATA_TYPE);
OMTensorList *(*dll_omTensorListCreate)(OMTensor **, int);
void (*dll_omTensorListDestroy)(OMTensorList *);

#if LOAD_MODEL_STATICALLY
#define RUN_MAIN_GRAPH run_main_graph
#define OM_INPUT_SIGNATURE omInputSignature
#define OM_OUTPUT_SIGNATURE omOutputSignature
#define OM_TENSOR_CREATE omTensorCreate
#define OM_TENSOR_LIST_CREATE omTensorListCreate
#define OM_TENSOR_LIST_DESTROY omTensorListDestroy
#define OPTIONS "hn:m:v"
#else
#define RUN_MAIN_GRAPH dll_run_main_graph
#define OM_INPUT_SIGNATURE dll_omInputSignature
#define OM_OUTPUT_SIGNATURE dll_omOutputSignature
#define OM_TENSOR_CREATE dll_omTensorCreate
#define OM_TENSOR_LIST_CREATE dll_omTensorListCreate
#define OM_TENSOR_LIST_DESTROY dll_omTensorListDestroy
#define OPTIONS "e:hn:m:v"
#endif

static int sIterations = 1;
static bool verbose = false;
static bool measureExecTime = false;

void usage(const char *name) {
#if LOAD_MODEL_STATICALLY
  cout << "Usage: " << name << " [options]";
#else
  cout << "Usage: " << name << " [options] model.so";
#endif
  cout << endl << endl;
  cout << "  Program will instantiate the model given by \"model.so\"" << endl;
  cout << "  with random inputs, launch the computation, and ignore" << endl;
  cout << "  the results. A model is typically generated by lowering" << endl;
  cout << "  an ONNX model using a \"onnx-mlir --EmitLib model.onnx\"" << endl;
  cout << "  command. When the input model is not found as is, the" << endl;
  cout << "  path to the local directory is also prepended." << endl;
  cout << endl;
  cout << "  Options:" << endl;
#if !LOAD_MODEL_STATICALLY
  cout << "    -e name | --entry-point name" << endl;
  cout << "         Name of the ONNX model entry point." << endl;
  cout << "         Default is \"run_main_graph\"." << endl;
#endif
  cout << "    -n NUM | --iterations NUM" << endl;
  cout << "         Number of times to run the tests, default 1." << endl;
  cout << "    -m NUM | --meas NUM" << endl;
  cout << "         Measure the kernel execution time NUM times." << endl;
  cout << "         Min 5 iters, shortest/longest points dropped." << endl;
  cout << "    -v | --verbose" << endl;
  cout << "         Print the shape of the inputs and outputs." << endl;
  cout << "    -h | --help" << endl;
  cout << "         Print help message." << endl;
  cout << endl;
}

void loadDLL(string name, string entryPointName) {
  cout << "Load model file " << name << " with entry point " << entryPointName
       << endl;
  void *handle = dlopen(name.c_str(), RTLD_LAZY);
  if (!handle) {
    string qualifiedName = "./" + name;
    cout << "  Did not find model, try in current dir " << qualifiedName
         << endl;
    handle = dlopen(qualifiedName.c_str(), RTLD_LAZY);
  }
  assert(handle && "Error loading the model's dll file; you may have provide a "
                   "fully qualified path");
  dll_run_main_graph = (OMTensorList * (*)(OMTensorList *))
      dlsym(handle, entryPointName.c_str());
  assert(!dlerror() && "failed to load entry point");
  dll_omInputSignature = (const char *(*)())dlsym(handle, "omInputSignature");
  assert(!dlerror() && "failed to load omInputSignature");
  dll_omOutputSignature = (const char *(*)())dlsym(handle, "omOutputSignature");
  assert(!dlerror() && "failed to load omOutputSignature");
  dll_omTensorCreate =
      (OMTensor * (*)(void *, int64_t *, int64_t, OM_DATA_TYPE))
          dlsym(handle, "omTensorCreate");
  assert(!dlerror() && "failed to load omTensorCreate");
  dll_omTensorListCreate = (OMTensorList * (*)(OMTensor **, int))
      dlsym(handle, "omTensorListCreate");
  assert(!dlerror() && "failed to load omTensorListCreate");
  dll_omTensorListDestroy =
      (void (*)(OMTensorList *))dlsym(handle, "omTensorListDestroy");
  assert(!dlerror() && "failed to load omTensorListDestroy");
}

void parseArgs(int argc, char **argv) {
  int c;
  string entryPointName("run_main_graph");
  static struct option long_options[] = {
      {"entry-point", required_argument, 0, 'e'}, // Entry point.
      {"help", no_argument, 0, 'h'},              // Help.
      {"iterations", required_argument, 0, 'n'},  // Number of iterations.
      {"meas", required_argument, 0, 'm'},        // Measurement of time.
      {"verbose", no_argument, 0, 'v'},           // Verbose.
      {0, 0, 0, 0}};

  while (true) {
    int index = 0;
    c = getopt_long(argc, argv, OPTIONS, long_options, &index);
    if (c == -1)
      break;
    switch (c) {
    case 0:
      break;
    case 'e':
      entryPointName = optarg;
      break;
    case 'n':
      sIterations = atoi(optarg);
      break;
    case 'm':
#ifdef WIN32
      cout << "Measurement option currently not available, ignore." << endl;
#else
      sIterations = atoi(optarg);
      measureExecTime = true;
#endif
      break;
    case 'v':
      verbose = true;
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  if (measureExecTime && sIterations < 5)
    sIterations = 5;

// Process the DLL.
#if LOAD_MODEL_STATICALLY
  if (optind < argc) {
    cout << "error: model.so was compiled in, cannot provide one now" << endl;
    usage(argv[0]);
    exit(1);
  }
#else
  if (optind == argc) {
    cout << "error: need one model.so dynamic library" << endl;
    usage(argv[0]);
    exit(1);
  } else if (optind + 1 == argc) {
    string name = argv[optind];
    loadDLL(name, entryPointName);
  } else {
    cout << "error: handle only one model.so dynamic library at a time" << endl;
    usage(argv[0]);
    exit(1);
  }
#endif
}

/**
 * \brief Create and initialize an OMTensorList from the signature of a model
 *
 * This function parse the signature of a ONNX compiled network, attached to the
 * binary via a .so and will scan the JSON signature for its input. For each
 * input in turn, it create a tensor of the proper type and shape. Data will be
 * either initialized (if dataPtrList is provided), allocated (if dataPtrList is
 * null and dataAlloc is set to true), or will otherwise be left empty. In case
 * of errors, a null pointer returned.
 *
 *
 * @param dataPtrList Pointer to a list of data pointers of the right size, as
 * determined by the signature.
 * @param allocData When no dataPtrList is provided, the this boolean variable
 * determine if data is to be allocated or not, using the sizes determined by
 * the signature.
 * @param trace If true, provide a printout of the signatures (input and
 * putput).
 * @return pointer to the TensorList just created, or null on error.
 */
OMTensorList *omTensorListCreateFromInputSignature(
    void **dataPtrList, bool dataAlloc, bool trace) {
  const char *sigIn = OM_INPUT_SIGNATURE();
  if (trace) {
    cout << "Model Input Signature " << (sigIn ? sigIn : "(empty)") << endl;
    const char *sigOut = OM_OUTPUT_SIGNATURE();
    cout << "Output signature: " << (sigOut ? sigOut : "(empty)") << endl;
  }
  if (!sigIn)
    return nullptr;

  // Create inputs.
  auto JSONInput = llvm::json::parse(sigIn);
  assert(JSONInput && "failed to parse json");
  auto JSONArray = JSONInput->getAsArray();
  assert(JSONArray && "failed to parse json as array");

  // Allocate array of inputs.
  int inputNum = JSONArray->size();
  assert(inputNum >= 0 && inputNum < 100 && "out of bound number of inputs");
  OMTensor **inputTensors = nullptr;
  if (inputNum > 0)
    inputTensors = (OMTensor **)malloc(inputNum * sizeof(OMTensor *));
  // Scan each input tensor
  for (int i = 0; i < inputNum; ++i) {
    auto JSONItem = (*JSONArray)[i].getAsObject();
    auto JSONItemType = JSONItem->getString("type");
    assert(JSONItemType && "failed to get type");
    auto type = JSONItemType.getValue();
    auto JSONDimArray = JSONItem->getArray("dims");
    int rank = JSONDimArray->size();
    assert(rank > 0 && rank < 100 && "rank is out bound");
    // Gather shape.
    int64_t shape[100];
    size_t size = 1;
    for (int d = 0; d < rank; ++d) {
      // auto JSONDimItem = (*JSONDimArray)[d].getAsInteger();
      // size_t dim = JSONDimItem->getInteger();
      auto JSONDimValue = (*JSONDimArray)[d].getAsInteger();
      assert(JSONDimValue && "failed to get value");
      size_t dim = JSONDimValue.getValue();
      shape[d] = dim;
      size *= dim;
    }
    // Create a randomly initialized tensor of the right shape.
    OMTensor *tensor = nullptr;
    if (type.equals("float") || type.equals("f32")) {
      float *data = nullptr;
      if (dataPtrList) {
        data = (float *)dataPtrList[i];
      } else if (dataAlloc) {
        data = new float[size];
        assert(data && "failed to allocate data");
      }
      tensor = OM_TENSOR_CREATE(data, shape, rank, ONNX_TYPE_FLOAT);
    }
    assert(tensor && "addd support for the desired type");
    // Add tensor to list.
    inputTensors[i] = tensor;
    if (trace) {
      cout << "Input " << i << ": tensor of " << type.str() << " with shape ";
      for (int d = 0; d < rank; ++d)
        cout << shape[d] << " ";
      cout << "and " << size << " elements" << endl;
    }
  }
  return OM_TENSOR_LIST_CREATE(inputTensors, inputNum);
}

void printTime(double avg, double std, double factor, string unit) {
  int s = timeLogInMicroSec.size();
  int m = s / 2;
  printf("@time, %s, median, %.1f, avg, %.1f, std, %.1f, min, %.1f, max, %.1f, "
         "sample, %d\n",
      unit.c_str(), (double)timeLogInMicroSec[m] / factor,
      (double)(avg / factor), (double)(std / factor),
      (double)timeLogInMicroSec[1] / factor,
      (double)timeLogInMicroSec[s - 2] / factor, s - 2);
}

#ifdef WIN32

inline void processStartTime() {}
inline void processStopTime() {}
void displayTime() {}

#else

inline void processStartTime() {
  if (!measureExecTime)
    return;
  gettimeofday(&startTime, NULL);
}

inline void processStopTime() {
  if (!measureExecTime)
    return;
  gettimeofday(&stopTime, NULL);
  timersub(&stopTime, &startTime, &result);
  uint64_t time =
      (uint64_t)result.tv_sec * 1000000ull + (uint64_t)result.tv_usec;
  timeLogInMicroSec.emplace_back(time);
}

void displayTime() {
  if (!measureExecTime)
    return;
  sort(timeLogInMicroSec.begin(), timeLogInMicroSec.end());
  int s = timeLogInMicroSec.size();
  double avg = 0;
  for (int i = 1; i < s - 1; ++i)
    avg += (double)timeLogInMicroSec[i];
  avg = avg / (s - 2);
  double std = 0;
  for (int i = 1; i < s - 1; ++i)
    std += ((double)timeLogInMicroSec[i] - avg) *
           ((double)timeLogInMicroSec[i] - avg);
  std = sqrt(std / (s - 2));
  printTime(avg, std, 1, "micro-second");
  if (avg >= 1e3) {
    printTime(avg, std, 1e3, "milli-second");
  }
  if (avg >= 1e6) {
    printTime(avg, std, 1e6, "second");
  }
}
#endif

int main(int argc, char **argv) {
  // Init args.
  parseArgs(argc, argv);
  // Init inputs.
  OMTensorList *tensorListIn =
      omTensorListCreateFromInputSignature(nullptr, true, verbose);
  assert(tensorListIn && "failed to scan signature");
  // Call the compiled onnx model function.
  cout << "Start computing " << sIterations << " iterations" << endl;
  for (int i = 0; i < sIterations; ++i) {
    OMTensorList *tensorListOut = nullptr;
    processStartTime();
    tensorListOut = RUN_MAIN_GRAPH(tensorListIn);
    processStopTime();
    if (tensorListOut)
      OM_TENSOR_LIST_DESTROY(tensorListOut);
    if (i > 0 && i % 10 == 0)
      cout << "  computed " << i << " iterations" << endl;
  }
  cout << "Finish computing " << sIterations << " iterations" << endl;
  displayTime();

  // Cleanup.
  OM_TENSOR_LIST_DESTROY(tensorListIn);
  return 0;
}
