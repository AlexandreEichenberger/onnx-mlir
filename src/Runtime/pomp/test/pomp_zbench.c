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
//   POMP_ZB_SECTIONS=1,2,3,4,5,6,7  Sections to run.
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
//           (first pomp)
//        B  one XOR activate word + counter join
//        C  one XOR activate word + per-worker done flags polled by leader
//        D  one XOR activate word + 4-ary tree join
//        E  4-ary tree wake + 4-ary tree join (similar to libomp)
//        F  per-worker epoch wake + per-worker assignment + done flags
//           (pomp today)
//        F' F, waking the workers in reverse order: the last woken is the
//           first one the leader polls, the worst case for its in-order scan
//        G  F, with the leader sweeping a mask of the workers not yet done
//           instead of waiting on each in turn
//        G' G, waking the workers in reverse order
//        H  F, with done bytes packed 8 to a line
//        I  F, with each worker's epoch, assignment and done flag on one line
//           (as pomp lays them out)
//        J  I, with a release store of the epoch instead of a fetch_add
//        K  J, prefetching every worker's line for store before the stores
//        L  J, waking the workers farthest from the leader first
//   5. Barrier designs, per barrier, with T threads:
//        central  counter + epoch word (pomp today)
//        tree+bc  4-ary tree gather + broadcast epoch word
//        tree+tr  4-ary tree gather + 4-ary tree release
//   By default only E, F, I and L are timed; -DPOMP_ZB_ALL_DESIGNS times all.
//   6. Anatomy of F, F', G', I, J, K, L regions: when the workers see the
//      wake-up, when the last one is done, and how long the leader's join
//      takes after that; which worker is last; and the wake loop alone, with
//      no worker spinning.
//   7. SMT interference: one thread computes while a partner waits on its
//      SMT sibling, on another core of its chip, or on another chip; the
//      partner is asleep, spins as pomp does, spins and yields as libomp
//      does, or spins with a backoff. The slowdown of the computing thread.
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

// A clock that is cheap to read and the same on every CPU (TOD clock on Z).
static inline uint64_t ticks(void) {
#if defined(__s390x__)
  uint64_t t;
  __asm__ volatile("stckf %0" : "=Q"(t) : : "cc");
  return t;
#elif defined(__x86_64__)
  uint32_t lo, hi;
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
  uint64_t t;
  __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(t));
  return t;
#else
  return (uint64_t)(now() * 1e9);
#endif
}

static double ns_per_tick;

static void calibrate_ticks(void) {
  double t0 = now();
  uint64_t k0 = ticks();
  while (now() - t0 < 0.02)
    ;
  ns_per_tick = (now() - t0) * 1e9 / (double)(ticks() - k0);
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

enum {
  M_EPOCH,
  M_XOR_CTR,
  M_XOR_FLAGS,
  M_XOR_TREE,
  M_TREE_TREE,
  M_EPOCH_FLAGS,
  M_EPOCH_FLAGS_REV,
  M_EPOCH_SWEEP,
  M_EPOCH_SWEEP_REV,
  M_EPOCH_BYTES,
  M_LINE,
  M_LINE_STORE,
  M_LINE_PREFETCH,
  M_LINE_FAR,
  NMODELS
};
static const char *model_name[NMODELS] = {
    "A epoch/worker + counter join (first pomp)",
    "B xor word + counter join",
    "C xor word + per-worker done flags",
    "D xor word + 4-ary tree join",
    "E 4-ary tree wake + tree join (libomp-like)",
    "F epoch/worker + done flags (pomp today)",
    "F' F, waking in reverse order",
    "G epoch/worker + done flags, sweep pending",
    "G' G, waking in reverse order",
    "H epoch/worker + done bytes, 8 per line",
    "I go+assignment+done on one line (as pomp)",
    "J I, release store instead of fetch_add",
    "K J, prefetch lines for store first",
    "L J, farthest workers woken first",
};

// The designs still timed by default: E for libomp's design, F for the earlier
// runs, I for pomp today, L for the candidate. The others are settled (see
// POMP_Z_Tuning.md); build with -DPOMP_ZB_ALL_DESIGNS to time them again.
#ifdef POMP_ZB_ALL_DESIGNS
static int timed(int m) { return m >= 0; }
#else
static int timed(int m) {
  return m == M_TREE_TREE || m == M_EPOCH_FLAGS || m == M_LINE ||
         m == M_LINE_FAR;
}
#endif

// Models I on: one line per worker.
static int is_line(int m) { return m >= M_LINE; }
// Models A and F to H: per-worker epoch wake and assignment.
static int is_epoch(int m) {
  return m == M_EPOCH || (m >= M_EPOCH_FLAGS && !is_line(m));
}

static line_t go_[MAXT], done_[MAXT], wake_[MAXT];
static struct {
  _Alignas(CL) void (*fn)(int);
  int tid;
} asg[MAXT];
static struct {
  _Alignas(CL) void (*fn)(int);
} team;
static struct {
  _Alignas(CL) _Atomic uint64_t go;
  void (*fn)(int);
  int tid;
  _Atomic uint64_t done;
} wl_[MAXT];
// Order in which the leader wakes the workers, and each worker's one-way
// latency from the leader (ns; for L).
static int wake_order[MAXT];
static double lat_[MAXT];
static int lat_known;
static line_t arrived, activate, stopf;
static line_t dbytes_[MAXT / 8];
static uint64_t dbytes_mask[MAXT / 8];
// Section 6: when each worker saw its wake-up and when it was done.
static struct {
  _Alignas(CL) uint64_t go, fin;
} stamp_[MAXT];
static int stamping;
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
    switch (is_epoch(model) ? M_EPOCH : is_line(model) ? M_LINE : model) {
    case M_EPOCH:
      while (atomic_load_explicit(&go_[w].v, memory_order_acquire) == r) {
        if (STOPPED())
          return NULL;
        relax();
      }
      break;
    case M_LINE:
      while (atomic_load_explicit(&wl_[w].go, memory_order_acquire) == r) {
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
    if (stamping)
      stamp_[w].go = ticks();
    fn = is_epoch(model) ? asg[w].fn : is_line(model) ? wl_[w].fn : team.fn;
    tid = is_epoch(model) ? asg[w].tid : is_line(model) ? wl_[w].tid : w;
    fn(tid);
    if (stamping)
      stamp_[w].fin = ticks();
    // Signal the end of the region.
    switch (model) {
    case M_EPOCH:
    case M_XOR_CTR:
      atomic_fetch_add_explicit(&arrived.v, 1, memory_order_release);
      break;
    case M_XOR_FLAGS:
    case M_EPOCH_FLAGS:
    case M_EPOCH_FLAGS_REV:
    case M_EPOCH_SWEEP:
    case M_EPOCH_SWEEP_REV:
      atomic_store_explicit(&done_[w].v, r, memory_order_release);
      break;
    case M_EPOCH_BYTES:
      __atomic_store_n((uint8_t *)&dbytes_[w / 8].v + w % 8, (uint8_t)r,
          __ATOMIC_RELEASE);
      break;
    case M_LINE:
    case M_LINE_STORE:
    case M_LINE_PREFETCH:
    case M_LINE_FAR:
      atomic_store_explicit(&wl_[w].done, r, memory_order_release);
      break;
    default:
      tree_join(w, r);
      atomic_store_explicit(&done_[w].v, r, memory_order_release);
    }
  }
}

static void fj_fork(uint64_t r) {
  if (is_line(model)) {
    if (model == M_LINE_PREFETCH)
      for (int i = 1; i < T; ++i)
        __builtin_prefetch(&wl_[wake_order[i]], 1, 3);
    for (int i = 1; i < T; ++i) {
      int w = wake_order[i];
      wl_[w].fn = body;
      wl_[w].tid = w;
      if (model == M_LINE)
        atomic_fetch_add(&wl_[w].go, 1);
      else
        atomic_store_explicit(&wl_[w].go, r, memory_order_release);
    }
    return;
  }
  switch (is_epoch(model) ? M_EPOCH : model) {
  case M_EPOCH: {
    int rev = model == M_EPOCH_FLAGS_REV || model == M_EPOCH_SWEEP_REV;
    for (int i = 1; i < T; ++i) {
      int w = rev ? T - i : i;
      asg[w].fn = body;
      asg[w].tid = w;
      atomic_fetch_add(&go_[w].v, 1);
    }
    break;
  }
  case M_TREE_TREE:
    team.fn = body;
    for (int c = 1; c <= 4 && c < T; ++c)
      atomic_store_explicit(&wake_[c].v, r, memory_order_release);
    break;
  default:
    team.fn = body;
    atomic_fetch_xor(&activate.v, fullmask);
  }
}

// Each done flag is read until it is seen done, and then never again.
static void sweep_join(uint64_t r) {
  uint64_t pending = fullmask;
  for (;;) {
    for (uint64_t p = pending; p; p &= p - 1) {
      int w = __builtin_ctzll(p);
      if (atomic_load_explicit(&done_[w].v, memory_order_acquire) == r)
        pending &= ~((uint64_t)1 << w);
    }
    if (!pending)
      return;
    relax();
  }
}

static void fj_join(uint64_t r) {
  switch (model) {
  case M_EPOCH:
  case M_XOR_CTR:
    while (atomic_load_explicit(&arrived.v, memory_order_acquire) !=
           r * (uint64_t)(T - 1))
      relax();
    break;
  case M_XOR_FLAGS:
  case M_EPOCH_FLAGS:
  case M_EPOCH_FLAGS_REV:
    for (int w = 1; w < T; ++w)
      while (atomic_load_explicit(&done_[w].v, memory_order_acquire) != r)
        relax();
    break;
  case M_EPOCH_SWEEP:
  case M_EPOCH_SWEEP_REV:
    sweep_join(r);
    break;
  case M_LINE:
  case M_LINE_STORE:
  case M_LINE_PREFETCH:
  case M_LINE_FAR:
    for (int w = 1; w < T; ++w)
      while (atomic_load_explicit(&wl_[w].done, memory_order_acquire) != r)
        relax();
    break;
  case M_EPOCH_BYTES: {
    uint64_t want = (r & 0xff) * 0x0101010101010101ull;
    for (int g = 0; g * 8 < T; ++g)
      while ((atomic_load_explicit(&dbytes_[g].v, memory_order_acquire) &
                 dbytes_mask[g]) != (want & dbytes_mask[g]))
        relax();
    break;
  }
  default:
    tree_join(0, r);
  }
}

static void fj_region(uint64_t r) {
  fj_fork(r);
  body(0);
  fj_join(r);
}

static void reset_lines(void) {
  for (int i = 0; i < MAXT; ++i) {
    atomic_store(&go_[i].v, 0);
    atomic_store(&done_[i].v, 0);
    atomic_store(&wake_[i].v, 0);
    atomic_store(&wl_[i].go, 0);
    atomic_store(&wl_[i].done, 0);
  }
  for (int g = 0; g < MAXT / 8; ++g) {
    union {
      uint64_t u;
      uint8_t b[8];
    } m = {0};
    for (int i = 0; i < 8; ++i)
      if (8 * g + i >= 1 && 8 * g + i < T)
        m.b[i] = 0xff;
    dbytes_mask[g] = m.u;
    atomic_store(&dbytes_[g].v, 0);
  }
  atomic_store(&arrived.v, 0);
  atomic_store(&activate.v, 0);
  atomic_store(&stopf.v, 0);
  atomic_store(&nready, 0);
}

static pthread_t fj_th[MAXT];

// One-way latency from the leader's CPU to each worker's, measured once.
static void measure_latencies(void) {
  long saved = pp_iters;
  if (lat_known)
    return;
  pp_iters = 20000;
  for (int w = 1; w < MAXT && w < ncpus; ++w)
    lat_[w] = pingpong(cpus[w]);
  pp_iters = saved;
  lat_known = 1;
}

// Workers 1 to T-1, in index order, or for L farthest first.
static void set_wake_order(void) {
  for (int i = 1; i < T; ++i)
    wake_order[i] = i;
  if (model != M_LINE_FAR)
    return;
  measure_latencies();
  for (int i = 2; i < T; ++i)
    for (int j = i; j > 1 && lat_[wake_order[j]] > lat_[wake_order[j - 1]];
         --j) {
      int x = wake_order[j];
      wake_order[j] = wake_order[j - 1];
      wake_order[j - 1] = x;
    }
}

static void fj_setup(int m, int t) {
  model = m;
  T = t;
  fullmask = (T >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << T) - 1)) & ~1ull;
  set_wake_order();
  reset_lines();
}

static void fj_start(int m, int t) {
  fj_setup(m, t);
  for (int w = 1; w < T; ++w)
    pthread_create(&fj_th[w], NULL, fj_worker, (void *)(intptr_t)w);
  while (atomic_load(&nready) != T - 1)
    relax();
}

static void fj_stop(void) {
  atomic_store(&stopf.v, 1);
  for (int w = 1; w < T; ++w)
    pthread_join(fj_th[w], NULL);
}

// Time per region in us.
static double fj_run(int m, int t, long iters) {
  uint64_t r = 0;
  double t0, el;
  fj_start(m, t);
  for (long i = 0; i < 2000; ++i)
    fj_region(++r);
  t0 = now();
  for (long i = 0; i < iters; ++i)
    fj_region(++r);
  el = now() - t0;
  fj_stop();
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
    if (!timed(m))
      continue;
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
// 6. Anatomy of a region
// -----------------------------------------------------------------------------

enum { A_FORK, A_WAKE1, A_WAKEN, A_DONE, A_TAIL, A_REGION, NA };

// The wake loop with no worker spinning, its lines in the leader's cache: the
// cost of its instructions alone (ns).
static double fork_alone(int m, int t, long iters) {
  uint64_t r = 0, t0;
  fj_setup(m, t);
  for (long i = 0; i < 2000; ++i)
    fj_fork(++r);
  t0 = ticks();
  for (long i = 0; i < iters; ++i)
    fj_fork(++r);
  return (double)(ticks() - t0) / iters * ns_per_tick;
}

static void fj_anatomy(int m, int t, long iters) {
  double sum[NA] = {0}, sumlast = 0;
  long first = 0, end = 0;
  uint64_t r = 0;
  stamping = 1;
  fj_start(m, t);
  for (long i = 0; i < 2000 + iters; ++i) {
    uint64_t t0, tf, tj, g1 = ~(uint64_t)0, gn = 0, fin = 0;
    int last = 1;
    t0 = ticks();
    fj_fork(++r);
    tf = ticks();
    body(0);
    fj_join(r);
    tj = ticks();
    if (i < 2000)
      continue;
    for (int w = 1; w < T; ++w) {
      uint64_t g = stamp_[w].go, f = stamp_[w].fin;
      g1 = g < g1 ? g : g1;
      gn = g > gn ? g : gn;
      if (f > fin) {
        fin = f;
        last = w;
      }
    }
    // Signed: the clocks of two CPUs may differ by a tick.
    sum[A_FORK] += (double)(int64_t)(tf - t0);
    sum[A_WAKE1] += (double)(int64_t)(g1 - t0);
    sum[A_WAKEN] += (double)(int64_t)(gn - t0);
    sum[A_DONE] += (double)(int64_t)(fin - t0);
    sum[A_TAIL] += (double)(int64_t)(tj - fin);
    sum[A_REGION] += (double)(int64_t)(tj - t0);
    sumlast += last;
    first += last == 1;
    end += last == T - 1;
  }
  fj_stop();
  stamping = 0;
  printf("  %-3.*s %4d", (int)strcspn(model_name[m], " "), model_name[m], t);
  for (int k = 0; k < NA; ++k)
    printf(" %8.0f", sum[k] / iters * ns_per_tick);
  printf(" | %6.1f %5.0f%% %5.0f%% | %6.0f\n", sumlast / iters,
      100.0 * first / iters, 100.0 * end / iters, fork_alone(m, t, iters));
  fflush(stdout);
}

static void bench_anatomy(const int *ts, int nts, long iters) {
  static const int ms[] = {M_EPOCH_FLAGS, M_EPOCH_FLAGS_REV, M_EPOCH_SWEEP_REV,
      M_LINE, M_LINE_STORE, M_LINE_PREFETCH, M_LINE_FAR};
  int nms = sizeof(ms) / sizeof(ms[0]), tmax = 0;
  printf("\n6. Anatomy of a region, body = %ld dummy iterations (ns after the "
         "leader starts\n   waking; mean of %ld regions)\n",
      work_iters, iters);
  printf("   fork: the leader's wake loop. woken 1st/last: first and last "
         "worker to see\n   its wake-up. done last: last worker's done store. "
         "tail: from there to the\n   end of the leader's join. last w: "
         "which worker is done last (mean), and how\n   often it is worker 1 "
         "(polled first) or T-1 (polled last).\n   alone: the wake loop with "
         "no worker spinning (its instructions only).\n");
  for (int i = 0; i < nts; ++i)
    tmax = ts[i] > tmax ? ts[i] : tmax;
  measure_latencies();
  printf("   one-way ns from the leader (CPU %d), workers 1 to %d:", cpus[0],
      tmax - 1);
  for (int w = 1; w < tmax; ++w)
    printf(" %.0f", lat_[w]);
  printf("\n");
  printf("  %-3s %4s %8s %8s %8s %8s %8s %8s | %6s %6s %6s | %6s\n", "", "T",
      "fork", "woken1st", "wokenlst", "donelast", "tail", "region", "last w",
      "w=1", "w=T-1", "alone");
  for (int j = 0; j < nms; ++j)
    for (int i = 0; i < nts; ++i)
      if (ts[i] > 1 && timed(ms[j]) && ms[j] != M_TREE_TREE)
        fj_anatomy(ms[j], ts[i], iters);
}

// -----------------------------------------------------------------------------
// 7. SMT interference
// -----------------------------------------------------------------------------

enum { P_ASLEEP, P_SPIN, P_SPIN_YIELD, P_SPIN_BACKOFF, NPMODES };
static const char *pmode_name[NPMODES] = {
    "asleep", "spin", "spin+yield", "spin+backoff"};
static line_t s7_stop;
static atomic_int s7_ready;
static int s7_mode;
static pthread_mutex_t s7_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t s7_cv = PTHREAD_COND_INITIALIZER;
static float s7_a[1024];

// A waiting thread: asleep, or polling a line that does not change, as a
// worker waits for its wake-up. spin+yield yields every 256 polls, about as
// often as libomp (KMP_USE_YIELD=1); spin+backoff waits between polls.
static void *s7_partner(void *arg) {
  pin((int)(intptr_t)arg);
  atomic_store(&s7_ready, 1);
  if (s7_mode == P_ASLEEP) {
    pthread_mutex_lock(&s7_mu);
    while (!atomic_load(&s7_stop.v))
      pthread_cond_wait(&s7_cv, &s7_mu);
    pthread_mutex_unlock(&s7_mu);
    return NULL;
  }
  for (long n = 0; !atomic_load_explicit(&s7_stop.v, memory_order_acquire);
       ++n) {
    relax();
    if (s7_mode == P_SPIN_YIELD && n % 256 == 255)
      sched_yield();
    else if (s7_mode == P_SPIN_BACKOFF)
      for (int k = 0; k < 64; ++k)
        __asm__ volatile("" ::: "memory");
  }
  return NULL;
}

static pthread_t s7_start(int mode, int cpu) {
  pthread_t th;
  s7_mode = mode;
  atomic_store(&s7_stop.v, 0);
  atomic_store(&s7_ready, 0);
  pthread_create(&th, NULL, s7_partner, (void *)(intptr_t)cpu);
  while (!atomic_load(&s7_ready))
    relax();
  return th;
}

static void s7_end(pthread_t th) {
  pthread_mutex_lock(&s7_mu);
  atomic_store(&s7_stop.v, 1);
  pthread_cond_broadcast(&s7_cv);
  pthread_mutex_unlock(&s7_mu);
  pthread_join(th, NULL);
}

// ns per 1024 elements: 0, the loop of pomp_test's `loop` (a[i] *= s); 1, a
// dependent chain of integer multiply-adds.
static double s7_kernel(int k, long reps) {
  volatile float vs = 1.0f;
  float s = vs;
  uint64_t x = 1;
  double t0 = now();
  for (long r = 0; r < reps; ++r) {
    if (k == 0) {
      for (int i = 0; i < 1024; ++i)
        s7_a[i] *= s;
      __asm__ volatile("" ::: "memory");
    } else {
      for (int i = 0; i < 1024; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        __asm__("" : "+r"(x));
      }
    }
  }
  sink += x;
  return (now() - t0) / reps * 1e9;
}

static int s7_same_chip(int a, int b) {
  return read_topo(a, "physical_package_id") ==
             read_topo(b, "physical_package_id") &&
         read_topo(a, "book_id") == read_topo(b, "book_id") &&
         read_topo(a, "drawer_id") == read_topo(b, "drawer_id");
}

static void bench_smt(void) {
  static const char *pname[3] = {"SMT sibling", "same chip", "other chip"};
  static const char *kname[2] = {"fp loop", "int chain"};
  int c0 = cpus[0], part[3] = {-1, -1, -1};
  long reps = 20000;
  printf("\n7. SMT interference: one thread computes on CPU %d while a "
         "partner waits\n   (ns per 1024 elements alone; then the time with "
         "the partner, as a ratio)\n",
      c0);
  if (!pinned || read_topo(c0, "core_id") < 0) {
    printf("  needs pinning and the Linux topology\n");
    return;
  }
  for (int i = 1; i < ncpus; ++i) {
    int c = cpus[i], chip = s7_same_chip(c0, c);
    int p = !chip ? 2
            : read_topo(c, "core_id") == read_topo(c0, "core_id") ? 0
                                                                   : 1;
    if (part[p] < 0)
      part[p] = c;
  }
  printf("  %-10s %-11s %4s %8s", "kernel", "partner", "cpu", "alone");
  for (int m = 0; m < NPMODES; ++m)
    printf(" %12s", pmode_name[m]);
  printf("\n");
  for (int k = 0; k < 2; ++k) {
    double v[REPS], alone;
    s7_kernel(k, reps / 10);
    for (int r = 0; r < REPS; ++r)
      v[r] = s7_kernel(k, reps);
    alone = median(v, REPS);
    for (int p = 0; p < 3; ++p) {
      if (part[p] < 0)
        continue;
      printf("  %-10s %-11s %4d %8.1f", kname[k], pname[p], part[p], alone);
      fflush(stdout);
      for (int m = 0; m < NPMODES; ++m) {
        for (int r = 0; r < REPS; ++r) {
          pthread_t th = s7_start(m, part[p]);
          s7_kernel(k, reps / 10);
          v[r] = s7_kernel(k, reps);
          s7_end(th);
        }
        printf(" %11.2fx", median(v, REPS) / alone);
        fflush(stdout);
      }
      printf("\n");
    }
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
  int sections = 0xfe;

  get_cpus();
  calibrate_ticks();
  if ((s = getenv("POMP_ZB_SECTIONS"))) {
    sections = 0;
    for (; *s; ++s)
      if (*s >= '1' && *s <= '7')
        sections |= 1 << (*s - '0');
  }
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

  if (sections & 1 << 1)
    bench_primitives();
  fflush(stdout);
  if (sections & 1 << 2)
    bench_pingpong();
  fflush(stdout);
  if (sections & 1 << 3)
    bench_contention(ts, nts);
  fflush(stdout);
  if (sections & 1 << 4)
    bench_forkjoin(ts, nts, iters);
  if (sections & 1 << 5)
    bench_barrier(ts, nts, iters);
  if (sections & 1 << 6)
    bench_anatomy(ts, nts, iters);
  fflush(stdout);
  if (sections & 1 << 7)
    bench_smt();
  printf("\ndone\n");
  return 0;
}
