/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- pomp_irbuilder_test.c --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Checks pomp with code from OpenMPIRBuilder, the code generator MLIR uses for
// onnx-mlir models, which also calls __kmpc_global_thread_num on the workers.
// Build with: clang -fopenmp -fopenmp-enable-irbuilder -I.. ... ../pomp.c
//
//===----------------------------------------------------------------------===//

#include "../pomp.h"
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// Like scf.parallel: omp parallel + omp for over an i64 index.
static long owner[100000];
void kernel(long n) {
#pragma omp parallel
  {
#pragma omp for
    for (long i = 0; i < n; ++i)
      owner[i] += 1 + i % 7;
  }
}
static void *donor(void *a) {
  pomp_donate_thread((int)(intptr_t)a);
  return 0;
}
int main(void) {
  pthread_t th[64];
  long n = 100000, sum = 0;
  if (pomp_init()) {
    fprintf(stderr, "init\n");
    return 1;
  }
  int L = pomp_get_thread_limit();
  for (int i = 1; i < L; ++i)
    pthread_create(&th[i], 0, donor, (void *)(intptr_t)i);
  while (pomp_num_donated() != L - 1)
    sched_yield();
  for (int r = 0; r < 1000; ++r)
    kernel(n);
  for (long i = 0; i < n; ++i)
    sum += owner[i];
  long expect = 0;
  for (long i = 0; i < n; ++i)
    expect += 1000L * (1 + i % 7);
  printf("irbuilder: sum %ld %s\n", sum, sum == expect ? "ok" : "BAD");
  pomp_release_workers();
  for (int i = 1; i < L; ++i)
    pthread_join(th[i], 0);
  return 0;
}
