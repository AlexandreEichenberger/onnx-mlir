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
//   imbal N LONG SHORT [join|barrier] [even|odd] : N regions in which the
//                even (or odd) tids run LONG passes of work and the others
//                SHORT, then meet at the join or at 10 barriers; prints the
//                time per phase (work and wait)
//   place      : empty fork/join regions, timed one by one, then traced:
//                where each thread runs, and the time of a region by how far
//                its farthest worker is from the leader (Linux)
//
// POMP_TEST_CPUS=0-7 (Linux) pins thread t of the run to the t-th CPU listed:
// the pool threads under pomp, and libomp's threads, in tid order, under
// libomp, so that both runtimes can be compared with the same placement.
//
//===----------------------------------------------------------------------===//

#ifdef __linux__
#define _GNU_SOURCE
#include <sched.h>
#include <sys/resource.h>
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
  int i, n = 0; // The whole thread limit.
  if (pomp_open_pool(&n) != POMP_OK) {
    fprintf(stderr, "open failed\n");
    exit(1);
  }
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
// place: where the threads of an empty fork/join region run, and how the
// time of a region depends on it. Two passes of iters regions each:
//   1. timed: each region timed alone, nothing else recorded;
//   2. traced: each thread records its CPU (sched_getcpu) in every region, and
//      each region is classified by where its farthest worker is from the
//      leader: on its core (an SMT sibling), on its chip, or elsewhere.
// Pass 2 is slowed by the trace (its cost per call is printed), so only the
// comparisons within it count. Also reports, from getrusage, the context
// switches of every team thread over pass 1, and the polarization of the CPUs
// used (Linux on Z).
// -----------------------------------------------------------------------------

#ifdef __linux__
#define PLACE_MAXCPU 1024
// Per CPU: core and chip ids, -2 if not read yet, -1 if unknown.
static int cpu_core[PLACE_MAXCPU], cpu_chip[PLACE_MAXCPU];
static char cpu_pol[PLACE_MAXCPU];

static int read_sysfs(int cpu, const char *name, char *buf, int len) {
  char path[256];
  FILE *f;
  int ok;
  snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/%s", cpu, name);
  if (!(f = fopen(path, "r")))
    return 0;
  ok = fgets(buf, len, f) != NULL;
  fclose(f);
  return ok;
}

static int read_topo_id(int cpu, const char *name) {
  char buf[64], path[64];
  snprintf(path, sizeof(path), "topology/%s", name);
  return read_sysfs(cpu, path, buf, sizeof(buf)) ? atoi(buf) : -1;
}

static void load_cpu(int c) {
  char buf[64];
  if (cpu_core[c] != -2)
    return;
  cpu_core[c] = read_topo_id(c, "core_id");
  // A chip: the same package in the same book and drawer.
  cpu_chip[c] = read_topo_id(c, "physical_package_id") +
                1000 * read_topo_id(c, "book_id") +
                1000000 * read_topo_id(c, "drawer_id");
  // vertical high / medium / low, or horizontal: h, m, l, z (horizontal).
  cpu_pol[c] = '?';
  if (read_sysfs(c, "polarization", buf, sizeof(buf)))
    cpu_pol[c] = !strncmp(buf, "vertical:high", 13)     ? 'h'
                 : !strncmp(buf, "vertical:medium", 15) ? 'm'
                 : !strncmp(buf, "vertical:low", 12)    ? 'l'
                 : !strncmp(buf, "horizontal", 10)      ? 'z'
                                                        : '?';
}

// 0: the same CPU, 1: an SMT sibling, 2: the same chip, 3: elsewhere.
static int distance(int a, int b) {
  if (a < 0 || b < 0 || a >= PLACE_MAXCPU || b >= PLACE_MAXCPU)
    return 3;
  load_cpu(a);
  load_cpu(b);
  if (a == b)
    return 0;
  if (cpu_chip[a] != cpu_chip[b])
    return 3;
  return cpu_core[a] == cpu_core[b] ? 1 : 2;
}

static int cmp_float(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return x < y ? -1 : x > y;
}

// Median, 90th and 99th percentiles of v[0..n) in us, sorting v.
static void pct(float *v, long n, float *p50, float *p90, float *p99) {
  qsort(v, n, sizeof(float), cmp_float);
  *p50 = n ? v[n / 2] : 0;
  *p90 = n ? v[n * 9 / 10] : 0;
  *p99 = n ? v[n * 99 / 100] : 0;
}

static void test_place(long iters) {
  int nth = num_threads;
  float *t1 = malloc(sizeof(float) * iters);
  float *t2 = malloc(sizeof(float) * iters);
  int *cpu = malloc(sizeof(int) * iters * nth);
  long csw0[64][2], csw1[64][2];
  long i, slow = 0;
  double sum = 0;
  for (i = 0; i < PLACE_MAXCPU; ++i)
    cpu_core[i] = cpu_chip[i] = -2;

  // Warm up.
  for (i = 0; i < 1000; ++i) {
#pragma omp parallel num_threads(nth)
    { __asm__ volatile("" ::: "memory"); }
  }
  // Pass 1, with each thread's context switches around it.
#pragma omp parallel num_threads(nth)
  {
    struct rusage ru;
    int t = omp_get_thread_num();
    getrusage(RUSAGE_THREAD, &ru);
    csw0[t][0] = ru.ru_nvcsw;
    csw0[t][1] = ru.ru_nivcsw;
  }
  for (i = 0; i < iters; ++i) {
    double a = now();
#pragma omp parallel num_threads(nth)
    { __asm__ volatile("" ::: "memory"); }
    t1[i] = (float)((now() - a) * 1e6);
    sum += t1[i];
  }
#pragma omp parallel num_threads(nth)
  {
    struct rusage ru;
    int t = omp_get_thread_num();
    getrusage(RUSAGE_THREAD, &ru);
    csw1[t][0] = ru.ru_nvcsw;
    csw1[t][1] = ru.ru_nivcsw;
  }

  // Pass 2.
  double c0 = now();
  for (i = 0; i < 100000; ++i)
    cpu[0] = sched_getcpu();
  double ccost = (now() - c0) / 100000 * 1e9;
  for (i = 0; i < iters; ++i) {
    double a = now();
#pragma omp parallel num_threads(nth)
    cpu[i * nth + omp_get_thread_num()] = sched_getcpu();
    t2[i] = (float)((now() - a) * 1e6);
  }

  // Pass 1: the distribution of region times.
  float p50, p90, p99;
  pct(t1, iters, &p50, &p90, &p99);
  for (i = 0; i < iters; ++i)
    slow += t1[i] > 2 * p50;
  printf("place: %d threads, %ld regions, sched_getcpu %.0f ns\n", nth, iters,
      ccost);
  printf("  timed:  us/region p50 %.3f p90 %.3f p99 %.3f mean %.3f;"
         " %.1f%% over 2x p50\n",
      p50, p90, p99, sum / iters, 100.0 * slow / iters);
  long v = 0, inv = 0;
  for (int t = 0; t < nth; ++t) {
    v += csw1[t][0] - csw0[t][0];
    inv += csw1[t][1] - csw0[t][1];
  }
  printf("  timed:  context switches per 1000 regions, all threads:"
         " voluntary %.2f involuntary %.2f\n",
      1000.0 * v / iters, 1000.0 * inv / iters);

  // Pass 2: where the threads ran.
  long far[4] = {0}, wd[4] = {0}, mig[2] = {0};
  float *byd[4];
  long nbyd[4] = {0};
  for (int d = 0; d < 4; ++d)
    byd[d] = malloc(sizeof(float) * iters);
  static char used[PLACE_MAXCPU];
  int ncpu = 0, npol[5] = {0};
  for (i = 0; i < iters; ++i) {
    int *c = &cpu[i * nth], fd = 0;
    for (int t = 0; t < nth; ++t) {
      if (c[t] >= 0 && c[t] < PLACE_MAXCPU && !used[c[t]]) {
        used[c[t]] = 1;
        ++ncpu;
      }
      if (i && c[t] != cpu[(i - 1) * nth + t])
        ++mig[t != 0];
      if (t) {
        int d = distance(c[0], c[t]);
        ++wd[d];
        fd = d > fd ? d : fd;
      }
    }
    ++far[fd];
    byd[fd][nbyd[fd]++] = t2[i];
  }
  for (int c = 0; c < PLACE_MAXCPU; ++c)
    if (used[c]) {
      load_cpu(c);
      const char *k = "hmlz?";
      ++npol[strchr(k, cpu_pol[c]) - k];
    }
  long nw = iters * (nth - 1);
  printf("  traced: %d CPUs used (polarization: %d vh, %d vm, %d vl, "
         "%d horizontal, %d unknown);\n"
         "          migrations per 1000 regions: leader %.2f, each worker "
         "%.2f\n",
      ncpu, npol[0], npol[1], npol[2], npol[3], npol[4],
      1000.0 * mig[0] / iters,
      nth > 1 ? 1000.0 * mig[1] / iters / (nth - 1) : 0.0);
  printf("  traced: workers from the leader: same CPU %.1f%%, SMT sibling "
         "%.1f%%, same chip %.1f%%, other chip %.1f%%\n",
      100.0 * wd[0] / nw, 100.0 * wd[1] / nw, 100.0 * wd[2] / nw,
      100.0 * wd[3] / nw);
  static const char *dn[4] = {"same CPU", "SMT sibling", "same chip",
      "other chip"};
  for (int d = 0; d < 4; ++d) {
    if (!nbyd[d])
      continue;
    pct(byd[d], nbyd[d], &p50, &p90, &p99);
    printf("  traced: farthest worker on %-11s %5.1f%% of regions,"
           " us/region p50 %.3f p90 %.3f p99 %.3f\n",
        dn[d], 100.0 * far[d] / iters, p50, p90, p99);
  }
  for (int d = 0; d < 4; ++d)
    free(byd[d]);
  free(t1);
  free(t2);
  free(cpu);
}
#else
static void test_place(long iters) {
  (void)iters;
  printf("place: Linux only\n");
}
#endif

// -----------------------------------------------------------------------------
// imbal: SMT interference at a join or a barrier. In each region the threads
// of even tid run `lng` passes of work and those of odd tid `shrt` (with
// `odd`, the reverse: the leader is short), then meet at the join, or at a
// barrier `phases` times per region. With POMP_TEST_CPUS listing both SMT
// threads of each core in turn, tids 2k and 2k+1 share a core: every long
// thread computes next to a short one that waits. Prints the time of a
// phase. Against the same run with one thread (the long work alone, its
// sibling idle), the excess is the runtime's overhead plus the slowdown of
// the long work by its waiting sibling.
// -----------------------------------------------------------------------------

static float imb_data[64][1024];

// One pass: the loop of `loop`, on 1024 floats in L1.
__attribute__((noinline)) static void imb_work(float *a, long passes, float s) {
  for (long p = 0; p < passes; ++p) {
    for (int i = 0; i < 1024; ++i)
      a[i] *= s;
    __asm__ volatile("" ::: "memory");
  }
}

static void test_imbal(long iters, long lng, long shrt, int barrier, int odd) {
  int phases = barrier ? 10 : 1;
  volatile float vs = 1.0f;
  float s = vs;
  double t0 = 0;
  long i;
  int nth = 0;
  for (i = 0; i < iters / 10 + 10 + iters; ++i) {
    if (i == iters / 10 + 10)
      t0 = now();
#pragma omp parallel
    {
      int t = omp_get_thread_num();
      long w = (t & 1) == odd ? lng : shrt;
      if (t == 0)
        nth = omp_get_num_threads();
      float *a = imb_data[t & 63];
      if (barrier) {
        for (int k = 0; k < phases; ++k) {
          imb_work(a, w, s);
#pragma omp barrier
        }
      } else {
        imb_work(a, w, s);
      }
    }
  }
  double phase = (now() - t0) / iters / phases;
  printf("imbal: %d threads, %s, long %ld short %ld passes on %s tids, "
         "phase %.3f us\n",
      nth, barrier ? "barrier" : "join", lng, shrt, odd ? "odd" : "even",
      phase * 1e6);
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

// With pomp built with -DPOMP_STATS: sleeps and wake-ups per region, over
// the whole run (warm-up included).
static void print_stats(void) {
#ifdef POMP
  unsigned long long c[7];
  if (pomp_get_stats(c, 7) < 7 || !c[0])
    return;
  double r = (double)c[0];
  printf("stats: %llu regions; per region: idle sleeps %.2f, team-wait sleeps "
         "%.2f, posts %.2f (at forks %.2f); per post %.0f ns posting; "
         "per sleep %.0f ns asleep\n",
      c[0], c[1] / r, c[2] / r, c[3] / r, c[4] / r,
      c[3] ? (double)c[5] / c[3] : 0.0,
      c[1] + c[2] ? (double)c[6] / (c[1] + c[2]) : 0.0);
#endif
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
  else if (!strcmp(test, "imbal"))
    test_imbal(n ? n : 1000, argc > 3 ? atol(argv[3]) : 2500,
        argc > 4 ? atol(argv[4]) : 250, argc > 5 && !strcmp(argv[5], "barrier"),
        argc > 6 && !strcmp(argv[6], "odd"));
  else if (!strcmp(test, "place"))
    test_place(n ? n : 100000);
  else if (!strcmp(test, "nestbench"))
    test_nestbench(n ? n : 100000, argc > 3 ? atoi(argv[3]) : 4,
        argc > 4 ? atoi(argv[4]) : 4);
  else {
    fprintf(stderr, "unknown test %s\n", test);
    return 1;
  }
  print_stats();
  release_all();
  return 0;
}
