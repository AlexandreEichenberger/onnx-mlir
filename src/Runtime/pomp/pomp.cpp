/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pomp.cpp ----------------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// pomp: a minimal pool-based OpenMP runtime. See POMP.md for the design.
//
// Implements the libomp entry points that onnx-mlir models (MLIR's
// OpenMPIRBuilder) and zDNNx (clang) emit:
//   __kmpc_global_thread_num   __kmpc_fork_call __kmpc_push_num_threads
//   __kmpc_push_proc_bind      __kmpc_serialized_parallel
//   __kmpc_end_serialized_parallel                       __kmpc_barrier
//   __kmpc_for_static_init_{4,4u,8,8u}                   __kmpc_for_static_fini
// plus a few omp_* queries.
//
// Model: any number of roots, host threads that run regions, each with its
// own descriptor (slot 0 of its teams), and up to 63 pool threads lent by the
// host (slots 1..limit-1). The 64-bit `busy` word has one bit per slot, clear
// iff that slot is an idle pool thread. A leader recruits its workers with one
// CAS on `busy`, and wakes each by bumping its epoch. At most two levels are
// active; deeper or empty forks run inline as a team of one. Every wait spins,
// then sleeps.
//
// Language: C++ for classes and templates only: no virtual functions,
// exceptions, RTTI, nor C++ headers, and constant-initialized globals. The
// object file needs only libc and pthreads, and links into C programs. See
// "Language" in POMP.md.
//
//===----------------------------------------------------------------------===//

#include "pomp.h"

#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

namespace {

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

// From pomp.h; at most 64, as slots are bits of one word.
constexpr int kMaxThreads = POMP_MAX_THREADS;
static_assert(kMaxThreads <= 64, "slots are bits of one 64-bit word");
// Unique ids of a team grow by this; every team index is below it (see
// "Unique ids" in pompSyncTypes.inc).
constexpr uint64_t kIdStride = uint64_t(1) << 16;

// Max shared args of an outlined fn (libomp's portable invoke allows 15).
constexpr int kMaxArgs = 16;

// Max nesting levels recorded; deeper ones are serialized and only counted.
constexpr int kMaxDepth = 8;
// Max nesting levels of any kind; deeper aborts.
constexpr int kMaxNesting = 256;
constexpr int kMaxActive = 2; // Max active (more than 1 thread) levels.
// How long a wait (idle, join, barrier) spins before it sleeps, in us. By
// default (POMP_SPIN=auto) each thread adapts it between kSpinFloorUs and the
// ceiling: on z, a thread spinning next to an SMT sibling that computes slows
// it by about 30%, so long waits should sleep at once; but a wake-up costs
// 2-10 us, so waits shorter than that must not sleep, or the next ones get
// longer and sleep too. See POMP_Z_Tuning.md, "Spin or sleep".
constexpr long kDefaultSpinMaxUs = 100;
constexpr long kSpinFloorUs = 2;
// Spin iterations between two reads of the clock.
constexpr long kSpinClockStride = 64;
constexpr long kYieldSpin = 4096; // spinUntil iterations before yield.

// Teams of at least this many threads use the tree barrier, smaller ones the
// central one (POMP_TREE_BARRIER overrides). On z17 the tree is 41-70% faster
// from 16 threads, on par at 8; on Apple M-series it is slower at every size.
// Unmeasured targets keep the central barrier. See POMP.md.
#if defined(__s390x__) || defined(__MVS__)
constexpr int kDefaultTreeBarrier = 5;
#else
constexpr int kDefaultTreeBarrier = kMaxThreads + 1; // Never.
#endif
constexpr int kTreeArity = 4;

// Size of a cache line, for alignment and padding.
#if defined(__s390x__) || defined(__MVS__)
constexpr int kCacheLine = 256;
#elif defined(__aarch64__) && defined(__APPLE__)
constexpr int kCacheLine = 128;
#else
constexpr int kCacheLine = 64;
#endif

// Schedule kinds (values of libomp's enum sched_type).
constexpr int32_t kSchStaticChunked = 33;
constexpr int32_t kSchStatic = 34;
constexpr int32_t kSchModifierMask = (1 << 29) | (1 << 30);

} // namespace

// Generic support code, in the same anonymous namespace; it uses kMaxArgs.
#include "pompSupport.inc"

namespace {

// pomp itself, in dependency order: not to be sorted. A class's declaration
// and code share a file, unless another class holds it by value
// (pompSyncTypes.inc, pompDescriptors.inc: declared before their holders,
// coded after); small methods live with their class; the code of a protocol
// (pool, roots, sync, region, loop) lives in its file. Code may call code of
// a later file: the whole unit is declared before any of it is compiled.
// clang-format off
#include "pompIdentity.inc"
#include "pompSettings.inc"
#include "pompSyncTypes.inc"
#include "pompDescriptors.inc"
#include "pompPool.inc"
#include "pompRoots.inc"
#include "pompSync.inc"
#include "pompRegion.inc"
#include "pompLoop.inc"
// clang-format on

// -----------------------------------------------------------------------------
// Initialization, at the first call of any entry point
// -----------------------------------------------------------------------------

// 0: not yet, 1: in progress, 2: done.
POMP_CONSTINIT Atomic<int> initState;

// Concurrent first calls wait for the one that does it.
void initOnce() {
  int expected = 0;
  if (initState.compareExchange(expected, 1)) {
    settings.init();
    pool.init();
    Self::createStorage();
    roots.init();
    initState.store(2, Order::Release);
  } else {
    while (initState.load(Order::Acquire) != 2)
      sched_yield();
  }
}

inline void ensureInit() {
  if (initState.load(Order::Acquire) != 2)
    initOnce();
}

} // namespace

// -----------------------------------------------------------------------------
// Host interface: the pool (pomp.h)
// -----------------------------------------------------------------------------

int pomp_get_thread_limit(void) {
  ensureInit();
  return settings.limit();
}

int pomp_open_pool(int fallback_limit, int *limit) {
  ensureInit();
  return pool.open(fallback_limit, limit);
}

int pomp_enter_pool(int slot) {
  ensureInit();
  return pool.enter(slot);
}

int pomp_close_pool(void) {
  ensureInit();
  return pool.close();
}

int pomp_wait_pool_empty(void) {
  ensureInit();
  return pool.waitEmpty();
}

int pomp_is_pool_open(void) {
  ensureInit();
  return pool.isOpen();
}

int pomp_pool_size(void) {
  ensureInit();
  return pool.size();
}

int pomp_set_option(const char *name, const char *value) {
  ensureInit();
  return settings.setOption(name, value, Settings::kCall);
}

int pomp_get_stats(unsigned long long *counts, int n) {
#ifdef POMP_STATS
  int k = n < kStats ? n : kStats;
  for (int i = 0; i < k; ++i)
    counts[i] = __atomic_load_n(&statCounters[i], __ATOMIC_RELAXED);
  return k;
#else
  (void)counts;
  (void)n;
  return 0;
#endif
}

// -----------------------------------------------------------------------------
// __kmpc entry points
// -----------------------------------------------------------------------------

extern "C" {

// Often the first call into pomp. Needs no init: before init, no thread has a
// descriptor, and a thread without one is slot 0, as any root.
int32_t __kmpc_global_thread_num(void *) {
  Thread *self = Thread::current();
  return self ? self->slot() : 0;
}

void __kmpc_push_num_threads(void *, int32_t, int32_t nth) {
  ensureInit();
  if (nth < 0)
    nth = 0;
  else if (nth > kMaxThreads)
    nth = kMaxThreads;
  Thread::currentOrNewRoot().setPending(nth);
}

void __kmpc_push_proc_bind(void *, int32_t, int32_t) {}

void __kmpc_fork_call(void *, int32_t argc, Microtask fn, ...) {
  void *argv[kMaxArgs];
  va_list ap;
  ensureInit();
  if (argc < 0)
    fatal("negative argument count for an outlined function");
  if (argc > kMaxArgs)
    fatal("too many arguments to an outlined function");
  va_start(ap, fn);
  for (int32_t i = 0; i < argc; ++i)
    argv[i] = va_arg(ap, void *);
  va_end(ap);
  forkCall(fn, argc, argv);
}

void __kmpc_serialized_parallel(void *, int32_t) {
  ensureInit();
  Thread &self = Thread::currentOrNewRoot();
  self.takePending();
  self.regions().pushSerialized();
}

void __kmpc_end_serialized_parallel(void *, int32_t) {
  Thread *self = Thread::current();
  if (!self)
    fatal("__kmpc_end_serialized_parallel without a matching "
          "__kmpc_serialized_parallel");
  self->regions().pop();
}

void __kmpc_barrier(void *, int32_t) {
  Thread *self = Thread::current();
  Level *innermost = self ? self->regions().activeInnermost() : nullptr;
  if (innermost)
    innermost->team->barrier(*self, *innermost);
}

void __kmpc_for_static_init_4(void *, int32_t, int32_t sched, int32_t *plast,
    int32_t *plower, int32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk) {
  staticInit<int32_t, int32_t, uint32_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_4u(void *, int32_t, int32_t sched, int32_t *plast,
    uint32_t *plower, uint32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk) {
  staticInit<uint32_t, int32_t, uint32_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_8(void *, int32_t, int32_t sched, int32_t *plast,
    int64_t *plower, int64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk) {
  staticInit<int64_t, int64_t, uint64_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_8u(void *, int32_t, int32_t sched, int32_t *plast,
    uint64_t *plower, uint64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk) {
  staticInit<uint64_t, int64_t, uint64_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_fini(void *, int32_t) {}

// -----------------------------------------------------------------------------
// omp_* queries
// -----------------------------------------------------------------------------

// Region queries need no init: before it, every thread is outside regions.
int omp_get_thread_num(void) {
  RegionStack *regions = Thread::currentRegions();
  return regions ? regions->innermostTid() : 0;
}

int omp_get_num_threads(void) {
  RegionStack *regions = Thread::currentRegions();
  return regions ? regions->innermostNth() : 1;
}

int omp_get_level(void) {
  RegionStack *regions = Thread::currentRegions();
  return regions ? regions->depth() : 0;
}

int omp_get_active_level(void) {
  RegionStack *regions = Thread::currentRegions();
  return regions ? regions->active() : 0;
}

int omp_in_parallel(void) { return omp_get_active_level() > 0; }

// Until a thread limit is set (pomp.h), these return 1 and warn once.
int omp_get_max_threads(void) {
  int active = omp_get_active_level();
  ensureInit();
  int n = settings.numThreads(active);
  if (n > 0)
    return n;
  settings.warnNoLimit();
  return 1;
}

int omp_get_thread_limit(void) {
  ensureInit();
  int n = settings.limit();
  if (n > 0)
    return n;
  settings.warnNoLimit();
  return 1;
}

int omp_get_max_active_levels(void) {
  ensureInit();
  return settings.maxActive();
}

} // extern "C"
