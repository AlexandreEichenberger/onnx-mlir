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
// worker threads lent through the pool by the test, as a host would) or with
// libomp, so that the printed results of the two can be compared line by line.
//
// Usage: pomp_test <test> [args]
//   static     : static loop partitioning, many bounds/steps/types/teams
//   nested     : team sizes of the zDNNx 3-level matmul nest and collapse
//   stress     : many regions with sleep/wake races (POMP_SPIN=0 helps)
//   barriers2  : barriers in level-1 and level-2 teams at once, of sizes that
//                change every region (needs OMP_MAX_ACTIVE_LEVELS=2)
//   lockstep   : workers that alternate between two level-2 teams whose
//                region and barrier counts stay equal (needs
//                OMP_MAX_ACTIVE_LEVELS=2)
//   cycles     : open / enter / run / close, repeated (the per-inference
//                pattern)
//   bench      : fork/join and barrier latency
//   loop       : one parallel loop region on a small array; prints the region,
//                the work of one thread's share alone, and their difference,
//                the runtime's overhead per region
//   nestbench N1 N2 : nested region, N1 outer threads each forking N2; prints
//                the region, work and overhead as `loop` does
//
// POMP_TEST_CPUS=0-7 (Linux) pins thread t of the run to the t-th CPU listed:
// the pool threads under pomp, and libomp's threads, in tid order, under
// libomp, so that both runtimes can be compared with the same placement.
//
//===----------------------------------------------------------------------===//

#ifdef __linux__
#define _GNU_SOURCE
#include <sched.h>
#endif
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
// Optional pinning (POMP_TEST_CPUS), the same for both runtimes
// -----------------------------------------------------------------------------

static int test_cpus[256], test_ncpus;
#ifdef __linux__
// Index in test_cpus of the CPU the calling thread is pinned to, or -1.
static __thread int pinned_index = -1;
#endif

static void parse_cpus(void) {
  const char *s = getenv("POMP_TEST_CPUS");
  while (s && *s && test_ncpus < 256) {
    char *e;
    long a = strtol(s, &e, 10), b = a;
    if (e == s)
      break;
    if (*e == '-')
      b = strtol(e + 1, &e, 10);
    for (long c = a; c <= b && test_ncpus < 256; ++c)
      test_cpus[test_ncpus++] = (int)c;
    s = *e == ',' ? e + 1 : e;
  }
}

// Pin the calling thread to the i-th listed CPU, if any.
static void pin_self(int i) {
#ifdef __linux__
  cpu_set_t set;
  if (i < 0 || i >= test_ncpus)
    return;
  CPU_ZERO(&set);
  CPU_SET(test_cpus[i], &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  pinned_index = i;
#else
  (void)i;
#endif
}

#ifndef POMP
// Pin the calling thread to the i-th listed CPU unless it already is: one
// thread-local compare when it is, which is how it stays inside a timing.
static inline void pin_once(int i) {
#ifdef __linux__
  if (test_ncpus && pinned_index != i)
    pin_self(i);
#else
  (void)i;
#endif
}
#endif

// -----------------------------------------------------------------------------
// Host side: threads entering the pool (stands for the Java layer)
// -----------------------------------------------------------------------------

#ifdef POMP
static pthread_t members[POMP_MAX_THREADS];

static void *member(void *arg) {
  int rc;
  pin_self((int)(intptr_t)arg);
  rc = pomp_enter_pool((int)(intptr_t)arg);
  if (rc != POMP_OK) {
    fprintf(stderr, "enter failed: %d\n", rc);
    exit(1);
  }
  return NULL;
}

static void enter_all(void) {
  int i, n = pomp_get_thread_limit();
  pomp_open_pool(0);
  for (i = 1; i < n; ++i)
    pthread_create(&members[i], NULL, member, (void *)(intptr_t)i);
  while (pomp_pool_size() != n - 1)
    sched_yield();
}

static void release_all(void) {
  int i, n = pomp_get_thread_limit();
  if (pomp_close_pool() != POMP_OK) {
    fprintf(stderr, "release failed\n");
    exit(1);
  }
  for (i = 1; i < n; ++i)
    pthread_join(members[i], NULL);
}
#else
static void enter_all(void) {}
static void release_all(void) {}
#endif

static void start(void) {
#ifdef POMP
  if (pomp_get_thread_limit() == 0) {
    fprintf(stderr, "no thread limit: set OMP_THREAD_LIMIT\n");
    exit(1);
  }
#endif
  num_threads = omp_get_thread_limit();
#ifndef POMP
  // Without OMP_THREAD_LIMIT, libomp's limit is INT_MAX: a team of that many
  // threads would exhaust memory. Use at most as many as pomp could.
  if (num_threads > 64)
    num_threads = 64;
#endif
  parse_cpus();
  pin_self(0);
  enter_all();
#ifndef POMP
  // libomp: pin each thread of a full team, which libomp keeps for later
  // teams, so tid t stays on the t-th CPU as pomp's slot t does. No more
  // threads than CPUs listed: extra spinning threads would share them.
  if (test_ncpus) {
    int n = num_threads < test_ncpus ? num_threads : test_ncpus;
#pragma omp parallel num_threads(n)
    pin_self(omp_get_thread_num());
  }
#endif
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
// barriers: several barriers per region, varying team sizes; each phase must
// see every write of the previous one.
// -----------------------------------------------------------------------------

static void test_barriers(long iters) {
  static long cell[64];
  long it, bad = 0;
  for (it = 0; it < iters; ++it) {
    int nth = 1 + (int)(it % num_threads), nbar = 1 + (int)(it % 5);
#pragma omp parallel num_threads(nth)
    {
      int t = omp_get_thread_num(), n = omp_get_num_threads();
      for (int b = 0; b < nbar; ++b) {
        __atomic_store_n(&cell[t], it * 8 + b, __ATOMIC_RELAXED);
#pragma omp barrier
        for (int i = 0; i < n; ++i)
          if (__atomic_load_n(&cell[i], __ATOMIC_RELAXED) != it * 8 + b)
            __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
#pragma omp barrier
      }
    }
  }
  printf("barriers: %ld regions, bad %ld\n", iters, bad);
}

// -----------------------------------------------------------------------------
// barriers2: two active levels. Each level-1 member leads a level-2 team; both
// levels pass barriers, interleaved, with team sizes, barrier counts and a
// late member that change every region, so that workers move between level-2
// teams and the leaders of level 2 meet level-1 barriers between their inner
// regions. Each phase must see every write of the previous one, at both
// levels.
// -----------------------------------------------------------------------------

static void test_barriers2(long iters) {
  static long cell1[64], cell2[64][64];
  long it, bad = 0, inactive = 0;
  for (it = 0; it < iters; ++it) {
    int n1 = 1 + (int)(it % (num_threads < 4 ? num_threads : 4));
    int nbar1 = 1 + (int)(it % 3);
#pragma omp parallel num_threads(n1)
    {
      int t1 = omp_get_thread_num(), m1 = omp_get_num_threads();
      for (int b1 = 0; b1 < nbar1; ++b1) {
        // Sizes differ between members and regions, from 1 to more than the
        // share, so that inner teams take workers from each other.
        int n2 = 1 + (int)((it * 7 + t1 * 3 + b1) % (num_threads / m1 + 2));
        int nbar2 = 1 + (int)((it + t1 + b1) % 4);
        long base = (it * 4 + b1) * 64;
#pragma omp parallel num_threads(n2)
        {
          int t2 = omp_get_thread_num(), m2 = omp_get_num_threads();
          if (n2 > 1 && m2 == 1)
            __atomic_fetch_add(&inactive, 1, __ATOMIC_RELAXED);
          for (int b2 = 0; b2 < nbar2; ++b2) {
            // A late member, so that others wait, and with POMP_SPIN=0 sleep.
            if (t2 == (int)((it + b2) % m2) && (it % 16) == 0)
              for (volatile int d = 0; d < 2000; ++d)
                ;
            __atomic_store_n(&cell2[t1][t2], base + b2, __ATOMIC_RELAXED);
#pragma omp barrier
            for (int i = 0; i < m2; ++i)
              if (__atomic_load_n(&cell2[t1][i], __ATOMIC_RELAXED) != base + b2)
                __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
#pragma omp barrier
          }
        }
        // Level 1, after the inner join: every member's inner team is done.
        __atomic_store_n(&cell1[t1], it * 4 + b1, __ATOMIC_RELAXED);
#pragma omp barrier
        for (int i = 0; i < m1; ++i)
          if (__atomic_load_n(&cell1[i], __ATOMIC_RELAXED) != it * 4 + b1)
            __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
#pragma omp barrier
      }
    }
  }
  printf("barriers2: %ld regions, bad %ld, inactive inner %ld\n", iters, bad,
      inactive);
}

// -----------------------------------------------------------------------------
// lockstep: a level-1 team of 2, whose members fork in turn a level-2 team of
// every other thread, with the same barriers. The two level-2 teams thus run
// the same number of regions and barriers, and the same workers serve one
// and then the other. A worker's sync line then holds, from the team it just
// left, the very counts that the other team waits for: only their tags tell
// them apart. Members other than the leader are late, so that a join or a
// barrier that a stale value lets pass is seen, as a member that is not done
// or a write that is missing.
// -----------------------------------------------------------------------------

static void test_lockstep(long iters) {
  static long cell[64], fin[64];
  long bad = 0, inactive = 0;
  int n2 = num_threads - 1;
  if (n2 < 2) {
    printf("lockstep: needs a limit of 3 or more\n");
    return;
  }
#pragma omp parallel num_threads(2)
  {
    int t1 = omp_get_thread_num();
    for (long it = 0; it < iters; ++it) {
      int nbar = 1 + (int)(it % 3);
      for (int turn = 0; turn < 2; ++turn) {
        long base = ((it * 2 + turn) * 4) * 64;
        if (t1 == turn) {
#pragma omp parallel num_threads(n2)
          {
            int t2 = omp_get_thread_num(), m2 = omp_get_num_threads();
            if (t2 == 0 && m2 != n2)
              __atomic_fetch_add(&inactive, 1, __ATOMIC_RELAXED);
            for (int b = 0; b < nbar; ++b) {
              if (t2 != 0)
                for (volatile int d = 0; d < 3000; ++d)
                  ;
              __atomic_store_n(&cell[t2], base + b, __ATOMIC_RELAXED);
#pragma omp barrier
              for (int i = 0; i < m2; ++i)
                if (__atomic_load_n(&cell[i], __ATOMIC_RELAXED) != base + b)
                  __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
#pragma omp barrier
            }
            if (t2 != 0)
              for (volatile int d = 0; d < 3000; ++d)
                ;
            __atomic_store_n(&fin[t2], base, __ATOMIC_RELAXED);
          }
          // After the join, every member must be done.
          for (int i = 0; i < n2; ++i)
            if (__atomic_load_n(&fin[i], __ATOMIC_RELAXED) != base)
              __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
        }
#pragma omp barrier
      }
    }
  }
  printf("lockstep: %ld cycles, bad %ld, short inner %ld\n", iters, bad,
      inactive);
}

// -----------------------------------------------------------------------------
// cycles: enter, run a few regions, close (once per "inference")
// -----------------------------------------------------------------------------

static void test_cycles(long cycles) {
  long c, bad = 0;
  double t0;
  release_all();
  t0 = now();
  for (c = 0; c < cycles; ++c) {
    int r;
    enter_all();
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
  enter_all();
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

// The work of one thread in a region: its share of the loop, as the
// worksharing loop runs it, without any runtime call. The barrier keeps the
// compiler from merging or removing the repeated calls.
__attribute__((noinline)) static void scale_share(
    float *a, long lo, long hi, float s) {
  for (long i = lo; i < hi; ++i)
    a[i] *= s;
  __asm__ volatile("" ::: "memory");
}

// Time per call of one thread's share of a loop of n over nth threads, the
// largest share of a static schedule: ceil(n / nth) elements.
static double share_time(float *a, long n, int nth, long iters) {
  long len = (n + nth - 1) / nth;
  volatile float vs = 1.0f;
  float s = vs;
  for (long i = 0; i < 1000; ++i)
    scale_share(a, 0, len, s);
  double t0 = now();
  for (long i = 0; i < iters; ++i)
    scale_share(a, 0, len, s);
  return (now() - t0) / iters;
}

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
  double region = (now() - t0) / iters;
  // The region lasts as long as its slowest thread's share: the overhead is
  // the rest, the runtime's cost of the region (fork, loop init, barrier,
  // join).
  double work = share_time(loop_data, n, num_threads, iters);
  printf("loop: %d threads, region %.3f us, work %.3f us, overhead %.3f us\n",
      num_threads, region * 1e6, work * 1e6, (region - work) * 1e6);
}

// -----------------------------------------------------------------------------
// nestbench: the zDNNx shape, an outer team of n1 whose members each fork an
// inner team of n2, then an inner loop and barrier. Needs
// OMP_MAX_ACTIVE_LEVELS=2 and a limit of at least n1 * n2.
// -----------------------------------------------------------------------------

static float nest_data[64][1024];

// Placement: with POMP_TEST_CPUS, inner thread i of outer member t goes on the
// CPU that pomp's pool gives it: t for i = 0 (the outer member itself), and
// n1 + t * (n2 - 1) + i - 1 otherwise, as the root splits the remaining slots
// into one contiguous group per outer member. pomp's threads are pinned that
// way as they enter the pool; libomp creates its inner threads from pinned
// outer members, whose CPU they would inherit, so under libomp they are
// pinned here, on their first region (and again only if libomp moves a thread
// between teams).
static void test_nestbench(long iters, int n1, int n2) {
  long i, short1 = 0, short2 = 0;
  double t0 = now();
  // Each outer member scales its own row by s, which is 1 but unknown to the
  // compiler, so the loop is not removed.
  volatile float vs = 1.0f;
  float s = vs;
  for (i = 0; i < 1000 + iters; ++i) {
    if (i == 1000)
      t0 = now();
#pragma omp parallel num_threads(n1)
    {
      float *a = nest_data[omp_get_thread_num() & 63];
      if (omp_get_thread_num() == 0 && omp_get_num_threads() != n1)
        __atomic_fetch_add(&short1, 1, __ATOMIC_RELAXED);
      int t = omp_get_thread_num();
      (void)t;
#pragma omp parallel num_threads(n2)
      {
#ifndef POMP
        int it = omp_get_thread_num();
        pin_once(it == 0 ? t : n1 + t * (n2 - 1) + it - 1);
#endif
#pragma omp for
        for (long k = 0; k < 1024; ++k)
          a[k] *= s;
        if (omp_get_thread_num() == 0 && omp_get_num_threads() != n2)
          __atomic_fetch_add(&short2, 1, __ATOMIC_RELAXED);
      }
    }
  }
  // A runtime that gives smaller teams than asked does less work: report it,
  // as a timing is then not comparable.
  double region = (now() - t0) / iters;
  double work = share_time(nest_data[0], 1024, n2, iters);
  printf("nestbench: %dx%d threads, short teams %ld outer %ld inner, region "
         "%.3f us, work %.3f us, overhead %.3f us\n",
      n1, n2, short1, short2, region * 1e6, work * 1e6,
      (region - work) * 1e6);
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
  else if (!strcmp(test, "barriers"))
    test_barriers(n ? n : 20000);
  else if (!strcmp(test, "barriers2"))
    test_barriers2(n ? n : 5000);
  else if (!strcmp(test, "lockstep"))
    test_lockstep(n ? n : 5000);
  else if (!strcmp(test, "cycles"))
    test_cycles(n ? n : 1000);
  else if (!strcmp(test, "loop"))
    test_loop(n ? n : 100000);
  else if (!strcmp(test, "bench"))
    test_bench(n ? n : 100000);
  else if (!strcmp(test, "nestbench"))
    test_nestbench(n ? n : 100000, argc > 3 ? atoi(argv[3]) : 4,
        argc > 4 ? atoi(argv[4]) : 4);
  else {
    fprintf(stderr, "unknown test %s\n", test);
    return 1;
  }
  release_all();
  return 0;
}
