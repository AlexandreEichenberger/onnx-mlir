/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- pomp_zbench.c ------------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Microbenchmarks to choose pomp's fork/join design on a target machine, in
// particular Linux on Z. Self-contained: C11 and pthreads, no OpenMP needed.
//
// Build (gcc or clang):
//   cc -O2 -std=gnu11 -march=native -pthread pomp_zbench.c -o pomp_zbench
// Check that atomics are single instructions (expect laxg, laog, laag, csg):
//   objdump -d pomp_zbench | grep -c -E 'laxg|laog|lang|laag'
//
// Run:
//   ./pomp_zbench                 # thread counts 2,4,8,16,32,64 (up to #CPUs)
//   ./pomp_zbench 4,8,16          # chosen thread counts
// Environment:
//   POMP_ZB_CPUS=0,2,4-11         CPUs to use, in order; thread t runs on the
//                                 t-th one. Default: all CPUs allowed.
//   POMP_ZB_ONE_PER_CORE=1        Use only the first SMT thread of each core.
//   POMP_ZB_ITERS=50000           Regions (and barriers) per measurement.
//   POMP_ZB_WORK=0                Dummy loop iterations per thread per region.
//
// Please send back the whole output, plus the output of `lscpu` and
// `lscpu -e`. Run on an otherwise idle LPAR if possible.
//
// What is measured:
//   1. Cost of each atomic operation, uncontended, on one thread.
//   2. Cache-line transfer latency between CPU 0 and every other CPU, with
//      each CPU's topology (core, socket, book, drawer), to map the machine.
//   3. Throughput of T threads incrementing one shared counter.
//   4. Fork/join designs, with spinning workers, per region:
//        A  per-worker epoch wake + per-worker assignment + counter join
//           (pomp today)
//        B  one XOR activate word + counter join
//        C  one XOR activate word + per-worker done flags polled by leader
//        D  one XOR activate word + 4-ary tree join
//        E  4-ary tree wake + 4-ary tree join (similar to libomp)
//   5. Barrier designs, per barrier, with T threads:
//        central  counter + epoch word (pomp today)
//        tree+bc  4-ary tree gather + broadcast epoch word
//        tree+tr  4-ary tree gather + 4-ary tree release
//
//===----------------------------------------------------------------------===//

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__s390x__)
#define CL 256
#elif defined(__aarch64__) && defined(__APPLE__)
#define CL 128
#else
#define CL 64
#endif
#define MAXT 64
#define MAXCPU 1024
#define REPS 3

typedef struct {
  _Alignas(CL) _Atomic uint64_t v;
} line_t;

static inline void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
  __asm__ volatile("yield" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static double median(double *v, int n) {
  qsort(v, n, sizeof(double), cmp_double);
  return v[n / 2];
}

// -----------------------------------------------------------------------------
// CPUs, pinning and topology
// -----------------------------------------------------------------------------

static int cpus[MAXCPU], ncpus;
static int pinned;

static int read_topo(int cpu, const char *name) {
#ifdef __linux__
  char path[256];
  FILE *f;
  int v = -1;
  snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu,
      name);
  f = fopen(path, "r");
  if (!f)
    return -1;
  if (fscanf(f, "%d", &v) != 1)
    v = -1;
  fclose(f);
  return v;
#else
  (void)cpu;
  (void)name;
  return -1;
#endif
}

// First CPU listed in this CPU's SMT sibling list.
static int first_sibling(int cpu) {
#ifdef __linux__
  char path[256];
  FILE *f;
  int v = cpu;
  snprintf(path, sizeof(path),
      "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
  f = fopen(path, "r");
  if (!f)
    return cpu;
  if (fscanf(f, "%d", &v) != 1)
    v = cpu;
  fclose(f);
  return v;
#else
  return cpu;
#endif
}

static void add_cpu(int c) {
  if (ncpus < MAXCPU)
    cpus[ncpus++] = c;
}

static void get_cpus(void) {
  const char *s = getenv("POMP_ZB_CPUS");
  if (s) {
    while (*s) {
      char *e;
      long a = strtol(s, &e, 10), b = a;
      if (e == s)
        break;
      if (*e == '-')
        b = strtol(e + 1, &e, 10);
      for (long c = a; c <= b; ++c)
        add_cpu((int)c);
      s = *e == ',' ? e + 1 : e;
    }
  } else {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    sched_getaffinity(0, sizeof(set), &set);
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &set))
        add_cpu(c);
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    for (int c = 0; c < n; ++c)
      add_cpu(c);
#endif
  }
  if (getenv("POMP_ZB_ONE_PER_CORE") && atoi(getenv("POMP_ZB_ONE_PER_CORE"))) {
    int n = 0;
    for (int i = 0; i < ncpus; ++i)
      if (first_sibling(cpus[i]) == cpus[i])
        cpus[n++] = cpus[i];
    ncpus = n;
  }
#ifdef __linux__
  pinned = 1;
#endif
}

static void pin(int cpu) {
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    pinned = 0;
#else
  (void)cpu;
#endif
}

// -----------------------------------------------------------------------------
// 1. Uncontended atomic operations
// -----------------------------------------------------------------------------

static line_t prim;
static volatile uint64_t sink;

#define PRIM(NAME, STMT)                                                       \
  do {                                                                         \
    uint64_t acc = 0;                                                          \
    double t0 = now();                                                         \
    for (long i = 0; i < n; ++i) {                                             \
      STMT;                                                                    \
    }                                                                          \
    double t = now() - t0;                                                     \
    sink += acc;                                                               \
    printf("  %-34s %8.2f ns\n", NAME, t / n * 1e9);                           \
  } while (0)

static void bench_primitives(void) {
  long n = 20000000;
  printf("\n1. Uncontended atomic operations on one 64-bit word (ns/op)\n");
  PRIM("load relaxed",
      acc += atomic_load_explicit(&prim.v, memory_order_relaxed));
  PRIM("load seq_cst", acc += atomic_load(&prim.v));
  PRIM("store release",
      atomic_store_explicit(&prim.v, (uint64_t)i, memory_order_release));
  PRIM("store seq_cst", atomic_store(&prim.v, (uint64_t)i));
  PRIM("fetch_add seq_cst", acc += atomic_fetch_add(&prim.v, 1));
  PRIM("fetch_add relaxed",
      acc += atomic_fetch_add_explicit(&prim.v, 1, memory_order_relaxed));
  PRIM("fetch_or seq_cst", acc += atomic_fetch_or(&prim.v, (uint64_t)i));
  PRIM("fetch_and seq_cst", acc += atomic_fetch_and(&prim.v, ~(uint64_t)i));
  PRIM("fetch_xor seq_cst", acc += atomic_fetch_xor(&prim.v, (uint64_t)i));
  PRIM("exchange seq_cst", acc += atomic_exchange(&prim.v, (uint64_t)i));
  PRIM("load + compare_exchange (succeeds)", {
    uint64_t e = atomic_load_explicit(&prim.v, memory_order_relaxed);
    acc += atomic_compare_exchange_strong(&prim.v, &e, e + 1);
  });
  PRIM("fence seq_cst (+ relaxed store)", {
    atomic_store_explicit(&prim.v, (uint64_t)i, memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
  });
}

// -----------------------------------------------------------------------------
// 2. Cache-line transfer latency (ping-pong on one line)
// -----------------------------------------------------------------------------

static line_t pp;
static atomic_int pp_ready;
static long pp_iters = 100000;

static void *pp_partner(void *arg) {
  pin((int)(intptr_t)arg);
  atomic_store(&pp_ready, 1);
  for (long i = 0; i < pp_iters; ++i) {
    while (atomic_load_explicit(&pp.v, memory_order_acquire) !=
           (uint64_t)(2 * i + 1))
      relax();
    atomic_store_explicit(&pp.v, (uint64_t)(2 * i + 2), memory_order_release);
  }
  return NULL;
}

// One-way latency in ns between the main thread's CPU and `cpu`.
static double pingpong(int cpu) {
  pthread_t th;
  double t0, t;
  atomic_store(&pp.v, 0);
  atomic_store(&pp_ready, 0);
  pthread_create(&th, NULL, pp_partner, (void *)(intptr_t)cpu);
  while (!atomic_load(&pp_ready))
    relax();
  t0 = now();
  for (long i = 0; i < pp_iters; ++i) {
    atomic_store_explicit(&pp.v, (uint64_t)(2 * i + 1), memory_order_release);
    while (atomic_load_explicit(&pp.v, memory_order_acquire) !=
           (uint64_t)(2 * i + 2))
      relax();
  }
  t = now() - t0;
  pthread_join(th, NULL);
  return t / pp_iters / 2 * 1e9;
}

static void print_topo_header(void) {
  printf("  %-5s %5s %6s %5s %6s", "cpu", "core", "socket", "book", "drawer");
}

static void print_topo(int c) {
  printf("  %-5d %5d %6d %5d %6d", c, read_topo(c, "core_id"),
      read_topo(c, "physical_package_id"), read_topo(c, "book_id"),
      read_topo(c, "drawer_id"));
}

static void bench_pingpong(void) {
  int step = ncpus > 65 ? (ncpus + 63) / 64 : 1;
  printf("\n2. Cache-line transfer latency, CPU %d <-> CPU k (one-way ns, "
         "half a round trip)\n",
      cpus[0]);
  print_topo_header();
  printf("  %10s\n", "one-way ns");
  print_topo(cpus[0]);
  printf("  %10s\n", "(self)");
  for (int k = 1; k < ncpus; k += step) {
    double v[REPS];
    for (int r = 0; r < REPS; ++r)
      v[r] = pingpong(cpus[k]);
    print_topo(cpus[k]);
    printf("  %10.1f\n", median(v, REPS));
  }
}

// -----------------------------------------------------------------------------
// 3. Contended fetch_add on one shared counter
// -----------------------------------------------------------------------------

static line_t ctr;
static atomic_int c_ready, c_go;
static long c_iters = 200000;

static void *c_worker(void *arg) {
  pin(cpus[(int)(intptr_t)arg]);
  atomic_fetch_add(&c_ready, 1);
  while (!atomic_load_explicit(&c_go, memory_order_acquire))
    relax();
  for (long i = 0; i < c_iters; ++i)
    atomic_fetch_add(&ctr.v, 1);
  return NULL;
}

static void bench_contention(const int *ts, int nts) {
  printf("\n3. T threads each doing fetch_add on one shared counter\n");
  printf("  %4s %16s %18s\n", "T", "ns/op (all)", "ns/op per thread");
  for (int i = 0; i < nts; ++i) {
    int T = ts[i];
    double v[REPS];
    for (int r = 0; r < REPS; ++r) {
      pthread_t th[MAXT];
      double t0;
      atomic_store(&ctr.v, 0);
      atomic_store(&c_ready, 0);
      atomic_store(&c_go, 0);
      for (int w = 1; w < T; ++w)
        pthread_create(&th[w], NULL, c_worker, (void *)(intptr_t)w);
      while (atomic_load(&c_ready) != T - 1)
        relax();
      t0 = now();
      atomic_store(&c_go, 1);
      for (long k = 0; k < c_iters; ++k)
        atomic_fetch_add(&ctr.v, 1);
      while (atomic_load(&ctr.v) != (uint64_t)T * c_iters)
        relax();
      v[r] = now() - t0;
      for (int w = 1; w < T; ++w)
        pthread_join(th[w], NULL);
    }
    double t = median(v, REPS);
    printf("  %4d %16.2f %18.2f\n", T, t / ((double)T * c_iters) * 1e9,
        t / c_iters * 1e9);
  }
}

// -----------------------------------------------------------------------------
// 4. Fork/join designs
// -----------------------------------------------------------------------------

enum { M_EPOCH, M_XOR_CTR, M_XOR_FLAGS, M_XOR_TREE, M_TREE_TREE, NMODELS };
static const char *model_name[NMODELS] = {
    "A epoch/worker + counter join (pomp today)",
    "B xor word + counter join",
    "C xor word + per-worker done flags",
    "D xor word + 4-ary tree join",
    "E 4-ary tree wake + tree join (libomp-like)",
};

static line_t go_[MAXT], done_[MAXT], wake_[MAXT];
static struct {
  _Alignas(CL) void (*fn)(int);
  int tid;
} asg[MAXT];
static struct {
  _Alignas(CL) void (*fn)(int);
} team;
static line_t arrived, activate, stopf;
static atomic_int nready;
static int T, model;
static uint64_t fullmask;
static long work_iters;

static void body(int tid) {
  volatile long x = 0;
  for (long i = 0; i < work_iters; ++i)
    x += i;
  (void)tid;
}

#define STOPPED() atomic_load_explicit(&stopf.v, memory_order_relaxed)

static void tree_join(int w, uint64_t r) {
  for (int c = 4 * w + 1; c <= 4 * w + 4 && c < T; ++c)
    while (atomic_load_explicit(&done_[c].v, memory_order_acquire) != r)
      relax();
}

static void *fj_worker(void *arg) {
  int w = (int)(intptr_t)arg;
  uint64_t r = 0, bit = (uint64_t)1 << w, expect = 0;
  pin(cpus[w]);
  atomic_fetch_add(&nready, 1);
  for (;;) {
    void (*fn)(int);
    int tid;
    // Wait for the next region.
    switch (model) {
    case M_EPOCH:
      while (atomic_load_explicit(&go_[w].v, memory_order_acquire) == r) {
        if (STOPPED())
          return NULL;
        relax();
      }
      break;
    case M_TREE_TREE:
      while (atomic_load_explicit(&wake_[w].v, memory_order_acquire) == r) {
        if (STOPPED())
          return NULL;
        relax();
      }
      for (int c = 4 * w + 1; c <= 4 * w + 4 && c < T; ++c)
        atomic_store_explicit(&wake_[c].v, r + 1, memory_order_release);
      break;
    default:
      while ((atomic_load_explicit(&activate.v, memory_order_acquire) & bit) ==
             expect) {
        if (STOPPED())
          return NULL;
        relax();
      }
      expect ^= bit;
    }
    ++r;
    fn = model == M_EPOCH ? asg[w].fn : team.fn;
    tid = model == M_EPOCH ? asg[w].tid : w;
    fn(tid);
    // Signal the end of the region.
    switch (model) {
    case M_EPOCH:
    case M_XOR_CTR:
      atomic_fetch_add_explicit(&arrived.v, 1, memory_order_release);
      break;
    case M_XOR_FLAGS:
      atomic_store_explicit(&done_[w].v, r, memory_order_release);
      break;
    default:
      tree_join(w, r);
      atomic_store_explicit(&done_[w].v, r, memory_order_release);
    }
  }
}

static void fj_region(uint64_t r) {
  switch (model) {
  case M_EPOCH:
    for (int w = 1; w < T; ++w) {
      asg[w].fn = body;
      asg[w].tid = w;
      atomic_fetch_add(&go_[w].v, 1);
    }
    break;
  case M_TREE_TREE:
    team.fn = body;
    for (int c = 1; c <= 4 && c < T; ++c)
      atomic_store_explicit(&wake_[c].v, r, memory_order_release);
    break;
  default:
    team.fn = body;
    atomic_fetch_xor(&activate.v, fullmask);
  }
  body(0);
  switch (model) {
  case M_EPOCH:
  case M_XOR_CTR:
    while (atomic_load_explicit(&arrived.v, memory_order_acquire) !=
           r * (uint64_t)(T - 1))
      relax();
    break;
  case M_XOR_FLAGS:
    for (int w = 1; w < T; ++w)
      while (atomic_load_explicit(&done_[w].v, memory_order_acquire) != r)
        relax();
    break;
  default:
    tree_join(0, r);
  }
}

static void reset_lines(void) {
  for (int i = 0; i < MAXT; ++i) {
    atomic_store(&go_[i].v, 0);
    atomic_store(&done_[i].v, 0);
    atomic_store(&wake_[i].v, 0);
  }
  atomic_store(&arrived.v, 0);
  atomic_store(&activate.v, 0);
  atomic_store(&stopf.v, 0);
  atomic_store(&nready, 0);
}

// Time per region in us.
static double fj_run(int m, int t, long iters) {
  pthread_t th[MAXT];
  uint64_t r = 0;
  double t0, el;
  model = m;
  T = t;
  fullmask = (T >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << T) - 1)) & ~1ull;
  reset_lines();
  for (int w = 1; w < T; ++w)
    pthread_create(&th[w], NULL, fj_worker, (void *)(intptr_t)w);
  while (atomic_load(&nready) != T - 1)
    relax();
  for (long i = 0; i < 2000; ++i)
    fj_region(++r);
  t0 = now();
  for (long i = 0; i < iters; ++i)
    fj_region(++r);
  el = now() - t0;
  atomic_store(&stopf.v, 1);
  for (int w = 1; w < T; ++w)
    pthread_join(th[w], NULL);
  return el / iters * 1e6;
}

static void bench_forkjoin(const int *ts, int nts, long iters) {
  printf("\n4. Fork/join per region, spinning workers, body = %ld dummy "
         "iterations (us/region)\n",
      work_iters);
  printf("  %-45s", "design \\ T");
  for (int i = 0; i < nts; ++i)
    printf(" %8d", ts[i]);
  printf("\n");
  for (int m = 0; m < NMODELS; ++m) {
    printf("  %-45s", model_name[m]);
    fflush(stdout);
    for (int i = 0; i < nts; ++i) {
      double v[REPS];
      for (int r = 0; r < REPS; ++r)
        v[r] = ts[i] > 1 ? fj_run(m, ts[i], iters) : 0;
      printf(" %8.3f", median(v, REPS));
      fflush(stdout);
    }
    printf("\n");
  }
}

// -----------------------------------------------------------------------------
// 5. Barrier designs
// -----------------------------------------------------------------------------

enum { B_CENTRAL, B_TREE_BCAST, B_TREE_TREE, NBARS };
static const char *bar_name[NBARS] = {
    "central counter + epoch (pomp today)",
    "4-ary tree gather + broadcast word",
    "4-ary tree gather + tree release",
};
static line_t bcount, bepoch, barr_[MAXT], brel_[MAXT], bstart;
static int bmodel;
static long biters;

static void barrier(int w, uint64_t k) {
  if (bmodel == B_CENTRAL) {
    if (atomic_fetch_add(&bcount.v, 1) == k * (uint64_t)T - 1)
      atomic_store_explicit(&bepoch.v, k, memory_order_release);
    else
      while (atomic_load_explicit(&bepoch.v, memory_order_acquire) != k)
        relax();
    return;
  }
  // Gather up the tree.
  for (int c = 4 * w + 1; c <= 4 * w + 4 && c < T; ++c)
    while (atomic_load_explicit(&barr_[c].v, memory_order_acquire) != k)
      relax();
  if (bmodel == B_TREE_BCAST) {
    if (w == 0)
      atomic_store_explicit(&bepoch.v, k, memory_order_release);
    else {
      atomic_store_explicit(&barr_[w].v, k, memory_order_release);
      while (atomic_load_explicit(&bepoch.v, memory_order_acquire) != k)
        relax();
    }
    return;
  }
  // Release down the tree.
  if (w != 0) {
    atomic_store_explicit(&barr_[w].v, k, memory_order_release);
    while (atomic_load_explicit(&brel_[w].v, memory_order_acquire) != k)
      relax();
  }
  for (int c = 4 * w + 1; c <= 4 * w + 4 && c < T; ++c)
    atomic_store_explicit(&brel_[c].v, k, memory_order_release);
}

static void *bar_worker(void *arg) {
  int w = (int)(intptr_t)arg;
  pin(cpus[w]);
  atomic_fetch_add(&nready, 1);
  while (!atomic_load_explicit(&bstart.v, memory_order_acquire))
    relax();
  for (long k = 1; k <= biters; ++k)
    barrier(w, (uint64_t)k);
  return NULL;
}

static double bar_run(int m, int t, long iters) {
  pthread_t th[MAXT];
  double t0, el;
  bmodel = m;
  T = t;
  biters = iters;
  atomic_store(&bcount.v, 0);
  atomic_store(&bepoch.v, 0);
  atomic_store(&bstart.v, 0);
  atomic_store(&nready, 0);
  for (int i = 0; i < MAXT; ++i) {
    atomic_store(&barr_[i].v, 0);
    atomic_store(&brel_[i].v, 0);
  }
  for (int w = 1; w < T; ++w)
    pthread_create(&th[w], NULL, bar_worker, (void *)(intptr_t)w);
  while (atomic_load(&nready) != T - 1)
    relax();
  t0 = now();
  atomic_store(&bstart.v, 1);
  for (long k = 1; k <= iters; ++k)
    barrier(0, (uint64_t)k);
  el = now() - t0;
  for (int w = 1; w < T; ++w)
    pthread_join(th[w], NULL);
  return el / iters * 1e6;
}

static void bench_barrier(const int *ts, int nts, long iters) {
  printf("\n5. Barrier, all T threads (us/barrier)\n");
  printf("  %-45s", "design \\ T");
  for (int i = 0; i < nts; ++i)
    printf(" %8d", ts[i]);
  printf("\n");
  for (int m = 0; m < NBARS; ++m) {
    printf("  %-45s", bar_name[m]);
    fflush(stdout);
    for (int i = 0; i < nts; ++i) {
      double v[REPS];
      for (int r = 0; r < REPS; ++r)
        v[r] = ts[i] > 1 ? bar_run(m, ts[i], iters) : 0;
      printf(" %8.3f", median(v, REPS));
      fflush(stdout);
    }
    printf("\n");
  }
}

// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  int ts[16], nts = 0;
  long iters = 50000;
  const char *s;

  get_cpus();
  if ((s = getenv("POMP_ZB_ITERS")))
    iters = atol(s);
  if ((s = getenv("POMP_ZB_WORK")))
    work_iters = atol(s);
  if (argc > 1) {
    for (char *p = argv[1]; *p && nts < 16;) {
      char *e;
      long v = strtol(p, &e, 10);
      if (e == p)
        break;
      if (v >= 1 && v <= MAXT && v <= ncpus)
        ts[nts++] = (int)v;
      p = *e == ',' ? e + 1 : e;
    }
  } else {
    static const int def[] = {2, 4, 8, 16, 32, 64};
    for (int i = 0; i < 6; ++i)
      if (def[i] <= ncpus)
        ts[nts++] = def[i];
  }
  pin(cpus[0]);

  printf("pomp_zbench\n");
#if defined(__s390x__)
  printf("  arch: s390x\n");
#elif defined(__aarch64__)
  printf("  arch: aarch64\n");
#elif defined(__x86_64__)
  printf("  arch: x86_64\n");
#endif
  printf("  compiler: %s\n", __VERSION__);
  printf("  cache line used for padding: %d bytes\n", CL);
  printf("  CPUs used (%d):", ncpus);
  for (int i = 0; i < ncpus; ++i)
    printf(" %d", cpus[i]);
  printf("\n  pinned: %s\n", pinned ? "yes" : "no (not Linux)");
  printf(
      "  regions/barriers per measurement: %ld, median of %d\n", iters, REPS);
  printf("  thread counts:");
  for (int i = 0; i < nts; ++i)
    printf(" %d", ts[i]);
  printf("\n");
  fflush(stdout);

  bench_primitives();
  fflush(stdout);
  bench_pingpong();
  fflush(stdout);
  bench_contention(ts, nts);
  fflush(stdout);
  bench_forkjoin(ts, nts, iters);
  bench_barrier(ts, nts, iters);
  printf("\ndone\n");
  return 0;
}
