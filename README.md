<!--- SPDX-License-Identifier: Apache-2.0 -->
<p align="center"><img width="50%" src="docs/logo/onnx-mlir-1280x640.png" /></p>

# ONNX MLIR
The Open Neural Network Exchange implementation in MLIR (http://onnx.ai/onnx-mlir/).

| System        | Build Status |
|---------------|--------------|
| s390x-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkins/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkins/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| ppc64le-Linux | [![Build Status](https://www.onnxmlir.xyz/jenkinp/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinp/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkinx/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Windows | [![Build Status](https://dev.azure.com/onnx-pipelines/onnx/_apis/build/status/MLIR-Windows-CI?branchName=main)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=9&branchName=main)             |
| amd64-macOS   | [![Build Status](https://github.com/onnx/onnx-mlir/workflows/Build%20x86%20onnx-mlir%20on%20macOS/badge.svg)](https://github.com/onnx/onnx-mlir/actions?query=workflow%3A%22Build+x86+onnx-mlir+on+macOS%22)             |

## Prebuilt Containers

The prefered approach to using and developing ONNX-MLIR is to used Docker Images and Containers, as getting the proper code dependences may be tricky on some systems. Our instructions on using ONNX-MLIR with dockers are [here](docs/Docker.md).

## Prerequisites

```
gcc >= 6.4
libprotoc >= 3.11.0
cmake >= 3.15.4
ninja >= 1.10.2
```
GCC can be found [here](https://gcc.gnu.org/install/), or if you have [Homebrew](https://docs.brew.sh/Installation), you can use `brew install gcc`. To check what version of gcc you have installed, run `gcc --version`.

The instructions to install libprotoc can be found [here](http://google.github.io/proto-lens/installing-protoc.htm). Or alternatively, if you have Homebrew, you can run `brew install protobuf`. To check what version you have installed, run `protoc --version`.

Cmake can be found [here](https://cmake.org/download/). However, to use Cmake, you need to follow the "How to Install For Command Line Use" tutorial, which can be found in Cmake under Tools>How to Install For Command Line Use. To check which version you have, you can either look in the desktop version under CMake>About, or run `cmake --version`.

The instructions for installing Ninja can be found [here](https://ninja-build.org/). Or, using Homebrew, you can run `brew install ninja`. To check the version, run `ninja --version`.



At any point in time, ONNX MLIR depends on a specific commit of the LLVM project that has been shown to work with the project. Periodically the maintainers
need to move to a more recent LLVM level. Among other things, this requires that the commit string in utils/clone-mlir.sh be updated. A consequence of
making this change is that the TravisCI build will fail until the Docker images that contain the prereqs are rebuilt. There is a GitHub workflow that rebuilds
this image for the amd64 architecture, but currently the ppc64le and s390x images must be rebuilt manually. The Dockerfiles to accomplish that are in the repo.

## Installation on UNIX

#### MLIR
Firstly, install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
``` bash
git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 0bf230d4220660af8b2667506f8905df2f716bdf && cd ..
```

[same-as-file]: <> (utils/build-mlir.sh)
``` bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

#### ONNX-MLIR (this project)
The following environment variables can be set before building onnx-mlir (or alternatively, they need to be passed as CMake variables):
- MLIR_DIR should point to the mlir cmake module inside an llvm-project build or install directory (e.g., llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](http://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running CMake, we can also specify the path to the lit tool from LLVM using the LLVM_EXTERNAL_LIT define but it is not required as long as MLIR_DIR points to a build directory of llvm-project. If MLIR_DIR points to an install directory of llvm-project, LLVM_EXTERNAL_LIT is required.

To build ONNX-MLIR, use the following commands:

[same-as-file]: <> ({"ref": "utils/install-onnx-mlir.sh", "skip-doc": 2})
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git

# Export environment variables pointing to LLVM-Projects.
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

mkdir onnx-mlir/build && cd onnx-mlir/build
cmake -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/c++ ..
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

If you are running on OSX Big Sur, you need to add `-DCMAKE_CXX_COMPILER=/usr/bin/c++`
to the `cmake ..` command due to changes in the compilers.
After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory.

##### LLVM and ONNX-MLIR CMake variables

The following CMake variables from LLVM and ONNX MLIR can be used when compiling ONNX MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

### MacOS Issues

There is a known issue when building onnx-mlir. If you see a error of this sorts
``` shell
Cloning into '/home/chentong/onnx-mlir/build/src/Runtime/jni/jsoniter'...

[...]

make[2]: *** [src/Runtime/jni/CMakeFiles/jsoniter.dir/build.make:74: src/Runtime/jni/jsoniter/target/jsoniter-0.9.23.jar] Error 127
make[1]: *** [CMakeFiles/Makefile2:3349: src/Runtime/jni/CMakeFiles/jsoniter.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
```

The suggested workaround before it's fixed: `brew install maven` and run `alias nproc="sysctl -n hw.logicalcpu"` in your shell.

## Installation on Windows
Building onnx-mlir on Windows requires building some additional prerequisites that are not available by default.

Note that the instructions in this file assume you are using [Visual Studio  2019 Community Edition](https://visualstudio.microsoft.com/downloads/) with ninja. It is recommended that you have the **Desktop development with C++** and **Linux development with C++** workloads installed. This ensures you have all toolchains and libraries needed to compile this project and its dependencies on Windows.

Run all the commands from a shell started from **"Developer Command Prompt for VS 2019"**.

#### Protobuf
Build protobuf as a static library.

[same-as-file]: <> (utils/install-protobuf.cmd)
```shell
git clone --recurse-submodules https://github.com/protocolbuffers/protobuf.git
REM Check out a specific branch that is known to work with ONNX MLIR.
REM This corresponds to the v3.11.4 tag
cd protobuf && git checkout d0bfd5221182da1a7cc280f3337b5e41a89539cf && cd ..

set root_dir=%cd%
md protobuf_build
cd protobuf_build
call cmake %root_dir%\protobuf\cmake -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf_install" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -Dprotobuf_BUILD_EXAMPLES=OFF ^
   -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
   -Dprotobuf_BUILD_TESTS=OFF ^
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
   -Dprotobuf_WITH_ZLIB=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
```

Before running CMake for onnx-mlir, ensure that the bin directory to this protobuf is before any others in your PATH:
```shell
set PATH=%root_dir%\protobuf_install\bin;%PATH%
```

#### MLIR
Install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
```shell
git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 0bf230d4220660af8b2667506f8905df2f716bdf && cd ..
```

[same-as-file]: <> (utils/build-mlir.cmd)
```shell
set root_dir=%cd%
md llvm-project\build
cd llvm-project\build
call cmake %root_dir%\llvm-project\llvm -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS=mlir ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF ^
   -DLLVM_INSTALL_UTILS=ON

call cmake --build . --config Release
call cmake --build . --config Release --target install
call cmake --build . --config Release --target check-mlir
```

#### ONNX-MLIR (this project)
The following environment variables can be set before building onnx-mlir (or alternatively, they need to be passed as CMake variables):
- MLIR_DIR should point to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](http://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running CMake, we can specify the path to the lit tool from LLVM using the LLVM_EXTERNAL_LIT define, as in the example below. If MLIR_DIR points to an install directory of llvm-project, LLVM_EXTERNAL_LIT is required and %lit_path% should point to a valid lit. It is not required if MLIR_DIR points to a build directory of llvm-project, which will contain lit.

To build ONNX MLIR, use the following commands:

[same-as-file]: <> ({"ref": "utils/build-onnx-mlir.cmd", "skip-doc": 2})
```shell
git clone --recursive https://github.com/onnx/onnx-mlir.git

set root_dir=%cd%

md onnx-mlir\build
cd onnx-mlir\build
call cmake %root_dir%\onnx-mlir -G "Ninja" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DCMAKE_PREFIX_PATH=%root_dir%\protobuf_install ^
   -DLLVM_EXTERNAL_LIT=%lit_path% ^
   -DLLVM_LIT_ARGS=-v ^
   -DMLIR_DIR=%root_dir%\llvm-project\build\lib\cmake\mlir

call cmake --build . --config Release --target onnx-mlir
```

To run the lit ONNX MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-mlir.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-lit
```

To run the numerical ONNX MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-numerical.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-numerical
```

To run the doc ONNX MLIR tests, use the following command after installing third_party ONNX:

[same-as-file]: <> ({"ref": "utils/check-docs.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-docs
```

After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory.

##### LLVM and ONNX-MLIR CMake variables

The following CMake variables from LLVM and ONNX MLIR can be used when compiling ONNX MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

## Using ONNX-MLIR

The usage of `onnx-mlir` is as such:
```
OVERVIEW: ONNX MLIR modular optimizer driver

USAGE: onnx-mlir [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNX MLIR Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXBasic - Ingest ONNX and emit the basic ONNX operations without inferred shapes.
      --EmitONNXIR    - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR      - Lower model to MLIR built-in transformation dialect.
      --EmitLLVMIR    - Lower model to LLVM IR (LLVM dialect).
      --EmitLib       - Lower model to LLVM IR, emit (to file) LLVM bitcode for model, compile and link it to a shared library.
```

## Simple Example

For example, to lower an ONNX model (e.g., add.onnx) to ONNX dialect, use the following command:
```shell
./onnx-mlir --EmitONNXIR add.onnx
```
The output should look like:
```mlir
module {
  func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```

An example based on the add operation is found [here](docs/doc_example), which build an ONNX model using a python script, and then provide a main program to load the model's value, compute, and print the models output.

## End to end example

An end to end example is provided [here](docs/mnist_example/README.md), which train, compile, and execute a simple MNIST example using both the C++ or Python interface.


## Troubleshooting

If the latest LLVM project fails to work due to the latest changes to the MLIR subproject please consider using a slightly older version of LLVM. One such version, which we use, can be found [here](https://github.com/clang-ykt/llvm-project).

## Installing `third_party ONNX` for Backend Tests or Rebuilding ONNX Operations

Backend tests are triggered by `make check-onnx-backend` in the build directory and require a few preliminary steps to run successfully. Similarily, rebuilding the ONNX operations in ONNX-MLIR from their ONNX descriptions is triggered by `make OMONNXOpsIncTranslation`.

You will need to install python 3.x if its not default in your environment, and possibly set the cmake `PYTHON_EXECUTABLE` varialbe in your top cmake file.

You will also need `pybind11` which may need to be installed (mac: `brew install pybind11` for example) and you may need to indicate where to find the software (Mac, POWER, possibly other platforms: `export pybind11_DIR=<your path to pybind>`). Then install the `third_party/onnx` software (Mac: `pip install -e third_party/onnx`) typed in the top directory.

On Macs/POWER and possibly other platforms, there is currently an issue that arises when installing ONNX. If you get an error during the build, try a fix where you edit the top CMakefile as reported in this PR: `https://github.com/onnx/onnx/pull/2482/files`.

While running `make check-onnx-backend` on a Mac you might encouter the following error: 

```shell
Fatal Python error: Aborted

Current thread 0x0000000107919e00 (most recent call first):
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2632 in getproxies_macosx_sysconf
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2650 in getproxies
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 795 in __init__
  ...
 ```

 A known workaround is to export the `no_proxy` environment variable in your shell as follow, and rerun the tests.

 ```shell
 % export no_proxy="*"
 ```

## Slack channel

We have a slack channel established under the Linux Foundation AI and Data Workspace, named `#onnx-mlir-discussion`. This channel can be used for asking quick questions related to this project. A direct link is [here](https://lfaifoundation.slack.com/archives/C01J4NAL4A2).

## Contributing

Want to contribute, consult this page for specific help on our project [here](CONTRIBUTING.md) or the docs sub-directory.
