# SPDX-License-Identifier: Apache-2.0

# pomp: a minimal pool-based OpenMP runtime, built as libompruntime.a in place
# of LLVM's libomp when ONNX_MLIR_USE_POMP is ON. See POMP.md.

if(NOT (OMP_SUPPORTED AND ONNX_MLIR_USE_POMP))
  return()
endif()

add_library(ompruntime STATIC pomp.c)
set_target_properties(ompruntime
  PROPERTIES
  LANGUAGE C
  C_STANDARD 11
  POSITION_INDEPENDENT_CODE TRUE
  ARCHIVE_OUTPUT_DIRECTORY ${ONNX_MLIR_LIBRARY_PATH}
  )
find_package(Threads REQUIRED)
target_link_libraries(ompruntime PUBLIC Threads::Threads)
# Use a pthread key instead of a __thread variable for the thread's slot.
if(POMP_USE_PTHREAD_KEY)
  target_compile_definitions(ompruntime PRIVATE POMP_USE_PTHREAD_KEY)
endif()

install(TARGETS ompruntime DESTINATION lib)
install(FILES pomp.h DESTINATION include)

if(ONNX_MLIR_BUILD_TESTS)
  # The test needs OpenMP pragmas compiled by a clang that has omp.h.
  add_executable(pomp_test test/pomp_test.c)
  target_compile_options(pomp_test PRIVATE -fopenmp)
  target_compile_definitions(pomp_test PRIVATE POMP)
  target_include_directories(pomp_test PRIVATE ${OPENMP_INCLUDE_DIR})
  target_link_libraries(pomp_test PRIVATE ompruntime)
  set_target_properties(pomp_test PROPERTIES EXCLUDE_FROM_ALL TRUE)

  # Host protocol scenarios (POMP.md); plain C, no OpenMP compiler needed.
  add_executable(pomp_host_test test/pomp_host_test.c)
  target_link_libraries(pomp_host_test PRIVATE ompruntime)
  set_target_properties(pomp_host_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
