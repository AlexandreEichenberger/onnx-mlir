# SPDX-License-Identifier: Apache-2.0

# pomp: a minimal pool-based OpenMP runtime, built as libompruntime.a in place
# of LLVM's libomp when ONNX_MLIR_USE_POMP is ON. See POMP.md.

if(NOT (OMP_SUPPORTED AND ONNX_MLIR_USE_POMP))
  return()
endif()

# pomp.cpp is C++ for its classes and templates only: no C++ header, nor any
# C++ runtime (exceptions, RTTI, thread-safe statics, new/delete). It needs
# what a C library needs, so it is built and linked as one: CMake sees a C
# source, compiled as C++ by the C compiler, and so adds no C++ standard
# library to the programs that link it, which a CXX source would make it do.
set_source_files_properties(pomp.cpp PROPERTIES LANGUAGE C)
add_library(ompruntime STATIC pomp.cpp)
set_target_properties(ompruntime
  PROPERTIES
  LINKER_LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  ARCHIVE_OUTPUT_DIRECTORY ${ONNX_MLIR_LIBRARY_PATH}
  )
# -x c++ and -std=c++17 come after any C standard flag CMake adds.
target_compile_options(ompruntime PRIVATE
  -x c++ -std=c++17
  -nostdinc++ -fno-exceptions -fno-rtti -fno-threadsafe-statics
  -funwind-tables)
find_package(Threads REQUIRED)
target_link_libraries(ompruntime PUBLIC Threads::Threads)
# Use a pthread key instead of a __thread variable for the thread's slot.
if(POMP_USE_PTHREAD_KEY)
  target_compile_definitions(ompruntime PRIVATE POMP_USE_PTHREAD_KEY)
endif()

install(TARGETS ompruntime DESTINATION lib)
install(FILES pomp.h DESTINATION include)

if(ONNX_MLIR_BUILD_TESTS)
  # The tests are C programs, linked with the C driver and without any C++
  # library: a C++ runtime symbol that pomp would need is a link error.

  # The test needs OpenMP pragmas compiled by a clang that has omp.h.
  add_executable(pomp_test test/pomp_test.c)
  target_compile_options(pomp_test PRIVATE -fopenmp)
  target_compile_definitions(pomp_test PRIVATE POMP)
  target_include_directories(pomp_test PRIVATE ${OPENMP_INCLUDE_DIR})
  target_link_libraries(pomp_test PRIVATE ompruntime)
  set_target_properties(pomp_test PROPERTIES
    LINKER_LANGUAGE C
    EXCLUDE_FROM_ALL TRUE)

  # Host protocol scenarios (POMP.md); plain C, no OpenMP compiler needed.
  add_executable(pomp_host_test test/pomp_host_test.c)
  target_link_libraries(pomp_host_test PRIVATE ompruntime)
  set_target_properties(pomp_host_test PROPERTIES
    LINKER_LANGUAGE C
    EXCLUDE_FROM_ALL TRUE)
endif()
