/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- pomp_test.c --------------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Tests for pomp. Built with -fopenmp and linked either with pomp (-DPOMP,
// worker threads donated by the test as a host would) or with libomp, so that
// the printed results of the two can be compared line by line.
//
// Usage: pomp_test <test> [args]
//   static     : static loop partitioning, many bounds/steps/types/teams
//   nested     : team sizes of the zDNNx 3-level matmul nest and collapse
//   stress     : many regions with sleep/wake races (POMP_SPIN=0 helps)
//   cycles     : donate / run / release, repeated (the per-inference pattern)
//   bench      : fork/join and barrier latency
//
//===----------------------------------------------------------------------===//

#include <inttypes.h>
#include <omp.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef POMP
#include "../pomp.h"
#endif

static int num_threads = 4;

// -----------------------------------------------------------------------------
// Host side: donate threads (stands for the Java layer)
// -----------------------------------------------------------------------------

#ifdef POMP
static pthread_t donors[POMP_MAX_THREADS];

static void *donor(void *arg) {
  int rc = pomp_donate_thread((int)(intptr_t)arg);
  if (rc != POMP_OK) {
    fprintf(stderr, "donate failed: %d\n", rc);
    exit(1);
  }
  return NULL;
}

static void donate_all(void) {
  int i, n = pomp_get_thread_limit();
  pomp_open_workers();
  for (i = 1; i < n; ++i)
    pthread_create(&donors[i], NULL, donor, (void *)(intptr_t)i);
  while (pomp_num_donated() != n - 1)
    sched_yield();
}

static void release_all(void) {
  int i, n = pomp_get_thread_limit();
  if (pomp_release_workers() != POMP_OK) {
    fprintf(stderr, "release failed\n");
    exit(1);
  }
  for (i = 1; i < n; ++i)
    pthread_join(donors[i], NULL);
}
#else
static void donate_all(void) {}
static void release_all(void) {}
#endif

static void start(void) {
#ifdef POMP
  if (pomp_init() != POMP_OK) {
    fprintf(stderr, "pomp_init failed: OMP_THREAD_LIMIT\n");
    exit(1);
  }
#endif
  num_threads = omp_get_thread_limit();
  donate_all();
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// -----------------------------------------------------------------------------
// static: record which thread runs each iteration, print a digest
// -----------------------------------------------------------------------------

#define MAXIT 1000
static int owner[MAXIT];
static int count[MAXIT];

static void check_loop(
    const char *what, long n, int nth, unsigned *digest, int *bad) {
  long i;
  for (i = 0; i < n; ++i) {
    if (count[i] != 1)
      ++*bad;
    *digest = *digest * 31u + (unsigned)owner[i] + 1u;
  }
  if (getenv("POMP_TEST_VERBOSE")) {
    printf("%s n=%ld nth=%d:", what, n, nth);
    for (i = 0; i < n && i < 40; ++i)
      printf(" %d", owner[i]);
    printf("\n");
  }
}

#define RUN_LOOP(T, LB, UB, ST, CMP, SCHED)                                    \
  do {                                                                         \
    memset(count, 0, sizeof(count));                                           \
    _Pragma("omp parallel num_threads(nth)") {                                 \
      _Pragma(SCHED) for (T i = (LB); i CMP(UB); i += (ST)) {                  \
        long k = (long)((i - (T)(LB)) / (T)(ST));                              \
        owner[k] = omp_get_thread_num();                                       \
        __atomic_fetch_add(&count[k], 1, __ATOMIC_RELAXED);                    \
      }                                                                        \
    }                                                                          \
  } while (0)

static void test_static(void) {
  static const long trips[] = {0, 1, 2, 3, 5, 7, 8, 13, 63, 64, 65, 100, 999};
  static const int steps[] = {1, 2, 3, 7};
  unsigned digest = 0;
  int bad = 0, t, s, nth;
  for (nth = 1; nth <= num_threads; ++nth)
    for (t = 0; t < (int)(sizeof(trips) / sizeof(trips[0])); ++t)
      for (s = 0; s < (int)(sizeof(steps) / sizeof(steps[0])); ++s) {
        long n = trips[t];
        int st = steps[s];
        // Signed 32, up.
        RUN_LOOP(int32_t, 5, 5 + n * st, st, <, "omp for schedule(static)");
        check_loop("i32", n, nth, &digest, &bad);
        // Unsigned 32, up.
        RUN_LOOP(uint32_t, 3u, 3u + (uint32_t)(n * st), (uint32_t)st, <,
            "omp for schedule(static)");
        check_loop("u32", n, nth, &digest, &bad);
        // Signed 64, down.
        RUN_LOOP(
            int64_t, (int64_t)n * st, 0, -st, >, "omp for schedule(static)");
        check_loop("i64d", n, nth, &digest, &bad);
        // Unsigned 64, up.
        RUN_LOOP(uint64_t, 7u, 7u + (uint64_t)(n * st), (uint64_t)st, <,
            "omp for schedule(static)");
        check_loop("u64", n, nth, &digest, &bad);
        // Chunked.
        RUN_LOOP(int64_t, 0, n * st, st, <, "omp for schedule(static, 3)");
        check_loop("i64c", n, nth, &digest, &bad);
      }
  printf("static: digest %08x bad %d\n", digest, bad);
}

// -----------------------------------------------------------------------------
// nested: the zDNNx matmul nest (if clauses) and collapse(4)
// -----------------------------------------------------------------------------

static void amax(int *p, int v) {
  int o = __atomic_load_n(p, __ATOMIC_RELAXED);
  while (v > o && !__atomic_compare_exchange_n(
                      p, &o, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
    ;
}

static void matmul_nest(uint32_t BT, uint32_t MT, uint32_t NT, bool sb, bool sm,
    bool sn, int teams[3], long *sum) {
  long s = 0;
  teams[0] = teams[1] = teams[2] = 1;
#pragma omp parallel for num_threads(BT) if (sb)
  for (uint32_t bs = 0; bs < BT; ++bs) {
    amax(&teams[0], omp_get_num_threads());
#pragma omp parallel for num_threads(MT) if (sm)
    for (uint32_t m = 0; m < MT; ++m) {
      amax(&teams[1], omp_get_num_threads());
#pragma omp parallel for num_threads(NT) if (sn)
      for (uint32_t n = 0; n < NT; ++n) {
        amax(&teams[2], omp_get_num_threads());
        __atomic_fetch_add(
            &s, (long)(bs * 10000 + m * 100 + n), __ATOMIC_RELAXED);
      }
    }
  }
  *sum = s;
}

static void test_nested(void) {
  struct {
    const char *name;
    uint32_t BT, MT, NT;
    bool sb, sm, sn;
  } cases[] = {
      {"m+n", 2, 4, 4, false, true, true},
      {"bs+n", 4, 1, 4, true, false, true},
      {"n", 1, 1, 4, false, false, true},
      {"bs", 4, 2, 2, true, false, false},
  };
  int c;
  for (c = 0; c < 4; ++c) {
    int teams[3];
    long sum;
    matmul_nest(cases[c].BT, cases[c].MT, cases[c].NT, cases[c].sb, cases[c].sm,
        cases[c].sn, teams, &sum);
    printf("nested %-5s teams %d/%d/%d sum %ld\n", cases[c].name, teams[0],
        teams[1], teams[2], sum);
  }
  {
    long s = 0;
    uint32_t a = 3, b = 2, cc = 5, d = 7;
#pragma omp parallel for num_threads(num_threads) collapse(4)
    for (uint32_t e4 = 0; e4 < a; ++e4)
      for (uint32_t e3 = 0; e3 < b; ++e3)
        for (uint32_t e2 = 0; e2 < cc; ++e2)
          for (uint32_t e1 = 0; e1 < d; ++e1)
            __atomic_fetch_add(&s, (long)(e4 * 1000 + e3 * 100 + e2 * 10 + e1),
                __ATOMIC_RELAXED);
    printf("collapse4 sum %ld\n", s);
  }
}

// -----------------------------------------------------------------------------
// stress: many regions, each with a barrier; check counts
// -----------------------------------------------------------------------------

static void test_stress(long iters) {
  long it, bad = 0;
  static long hits[64];
  for (it = 0; it < iters; ++it) {
    int nth = 1 + (int)(it % num_threads);
    long total = 0;
    memset(hits, 0, sizeof(hits));
#pragma omp parallel num_threads(nth)
    {
      int t = omp_get_thread_num();
      hits[t]++;
#pragma omp barrier
      if (t == 0) {
        int i;
        for (i = 0; i < omp_get_num_threads(); ++i)
          total += hits[i];
        if (total != omp_get_num_threads())
          __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
      }
    }
  }
  printf("stress: %ld regions, bad %ld\n", iters, bad);
}

// -----------------------------------------------------------------------------
// cycles: donate, run a few regions, release (once per "inference")
// -----------------------------------------------------------------------------

static void test_cycles(long cycles) {
  long c, bad = 0;
  double t0;
  release_all();
  t0 = now();
  for (c = 0; c < cycles; ++c) {
    int r;
    donate_all();
    for (r = 0; r < 10; ++r) {
      long s = 0;
#pragma omp parallel for
      for (long i = 0; i < 1000; ++i)
        __atomic_fetch_add(&s, i, __ATOMIC_RELAXED);
      if (s != 999L * 1000 / 2)
        ++bad;
    }
    release_all();
  }
  printf("cycles: %ld, bad %ld, %.1f us/cycle\n", cycles, bad,
      (now() - t0) / cycles * 1e6);
  donate_all();
}

// -----------------------------------------------------------------------------
// bench: empty fork/join, and barrier
// -----------------------------------------------------------------------------

static void test_bench(long iters) {
  long i;
  double t0, t1, t2;
  int nth = num_threads;
  // Warm up.
  for (i = 0; i < 1000; ++i) {
#pragma omp parallel num_threads(nth)
    { __asm__ volatile("" ::: "memory"); }
  }
  t0 = now();
  for (i = 0; i < iters; ++i) {
#pragma omp parallel num_threads(nth)
    { __asm__ volatile("" ::: "memory"); }
  }
  t1 = now();
#pragma omp parallel num_threads(nth)
  {
    for (long j = 0; j < iters; ++j) {
#pragma omp barrier
    }
  }
  t2 = now();
  printf("bench: %d threads, fork/join %.2f us, barrier %.2f us\n", nth,
      (t1 - t0) / iters * 1e6, (t2 - t1) / iters * 1e6);
}

// -----------------------------------------------------------------------------
// loop: the POMP.md example, a parallel region with one static loop and its
// barrier, on a small array: the per-region runtime cost.
// -----------------------------------------------------------------------------

static float loop_data[4096];

static void scale(float *a, long n, float s) {
#pragma omp parallel
  {
#pragma omp for
    for (long i = 0; i < n; ++i)
      a[i] *= s;
  }
}

static void test_loop(long iters) {
  long i, n = 1024;
  double t0;
  for (i = 0; i < 1000; ++i)
    scale(loop_data, n, 1.0f);
  t0 = now();
  for (i = 0; i < iters; ++i)
    scale(loop_data, n, 1.0f);
  printf("loop: %d threads, %.2f us per region\n", num_threads,
      (now() - t0) / iters * 1e6);
}

int main(int argc, char **argv) {
  const char *test = argc > 1 ? argv[1] : "static";
  long n = argc > 2 ? atol(argv[2]) : 0;
  start();
  if (!strcmp(test, "static"))
    test_static();
  else if (!strcmp(test, "nested"))
    test_nested();
  else if (!strcmp(test, "stress"))
    test_stress(n ? n : 100000);
  else if (!strcmp(test, "cycles"))
    test_cycles(n ? n : 1000);
  else if (!strcmp(test, "loop"))
    test_loop(n ? n : 100000);
  else if (!strcmp(test, "bench"))
    test_bench(n ? n : 100000);
  else {
    fprintf(stderr, "unknown test %s\n", test);
    return 1;
  }
  release_all();
  return 0;
}
