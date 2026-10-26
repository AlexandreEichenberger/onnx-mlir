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
// Model: one root (the initial thread, slot 0) and up to 63 worker threads
// lent by the host through the pool (slots 1..limit-1). One 64-bit word,
// `busy`, holds one bit per slot; a bit is clear iff that slot is an idle
// worker in the pool. A leader recruits a team with a single CAS on `busy`,
// hands each worker the work through its descriptor, wakes it by bumping its
// private epoch, and clears the team's bits again after the join. At most two
// levels are active; deeper or empty forks run inline as a team of one.
//
// The join polls one done flag per member, and barriers of more than four
// threads use a 4-ary tree; see POMP.md for the measurements behind both.
// Every wait, idle or in a team, spins and then sleeps.
//
// Release does not block the leader: it closes the pool, claims the idle
// workers and marks them in `released`. Spinning workers see their bit and
// leave by themselves; each leaving worker wakes up to two sleeping ones, so
// the wake-ups spread as a tree. See POMP.md for the scenarios supported.
//
// Language: C++ for its classes and templates only. No virtual functions,
// exceptions, RTTI, nor any C++ header; every global is constant-initialized
// and trivially destructible. The thread and team descriptors are allocated
// once, with calloc, when the thread limit is set, and never freed. The object
// file thus needs only the C library and pthreads, as a C one does, and links
// into C programs with the C driver. See "Language" in POMP.md.
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

namespace {

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

// Max thread defined in pomp.h, currently limited by the code scheme to 64.
constexpr int kMaxThreads = POMP_MAX_THREADS;
static_assert(kMaxThreads <= 64, "slots are bits of one 64-bit word");
// Teams: one led by each slot at level 2, and the root's at level 1.
constexpr int kTeams = kMaxThreads + 1;

// Max shared args passed to an outlined fn: one more than the 15 of libomp's
// portable kmp_invoke_microtask.cpp.
constexpr int kMaxArgs = 16;

// Max nesting levels recorded; deeper ones are serialized and only counted.
constexpr int kMaxDepth = 8;
// Max nesting levels of any kind, on a region stack or in a foreign one;
// deeper aborts. Far beyond any real use, and small enough to fit the
// foreign depth of SelfWord.
constexpr int kMaxNesting = 256;
constexpr int kMaxActive = 2; // Max active (more than 1 thread) levels.
// Spin iterations of any wait (idle, join, barrier) before it sleeps.
constexpr long kDefaultSpin = 200000;
constexpr long kYieldSpin = 4096; // Team-wait spin iterations before yield.

// Teams of at least this many threads use the tree barrier, smaller ones the
// central one (POMP_TREE_BARRIER overrides it). Measured: on z17 the tree is
// 41-70% faster from 16 threads on and on par at 8; z/OS, on the same
// hardware, takes the same default. On Apple M-series the tree is slower at
// every size measured (up to 16 threads, about the core count). Other targets
// are unmeasured and keep the central barrier. See POMP.md.
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

// pomp itself, in parts that follow each other: types first, then the code
// on them. See the comment at the top of each file.
#include "pompIdentity.inc"
#include "pompSyncTypes.inc"
#include "pompTypes.inc"
#include "pompInit.inc"
#include "pompPool.inc"
#include "pompSync.inc"
#include "pompRegion.inc"
#include "pompLoop.inc"

} // namespace

// -----------------------------------------------------------------------------
// Host interface: the pool (pomp.h)
// -----------------------------------------------------------------------------

int pomp_get_thread_limit(void) {
  settings.ensureInit();
  return settings.limit();
}

int pomp_open_pool(int thread_limit) {
  settings.ensureInit();
  return pool.open(thread_limit);
}

int pomp_enter_pool(int slot) {
  settings.ensureInit();
  return pool.enter(slot);
}

int pomp_close_pool(void) {
  settings.ensureInit();
  return pool.close();
}

int pomp_wait_pool_empty(void) {
  settings.ensureInit();
  return pool.waitEmpty();
}

int pomp_is_pool_open(void) {
  settings.ensureInit();
  return !pool.isClosed();
}

int pomp_pool_size(void) {
  settings.ensureInit();
  return pool.size();
}

// -----------------------------------------------------------------------------
// __kmpc entry points
// -----------------------------------------------------------------------------

extern "C" {

// May be the first call into pomp: compilers emit it in the host function,
// before push_num_threads, serialized_parallel or fork_call. No init needed:
// a thread without a slot is slot 0, and before init no thread has one (only
// pomp_enter_pool makes workers, and it inits first). Unlike libomp, which
// registers the calling thread here, this is just a load of the thread's word.
int32_t __kmpc_global_thread_num(void *) { return SelfWord::current().slot(); }

void __kmpc_fork_call(void *, int32_t argc, Microtask fn, ...) {
  void *argv[kMaxArgs];
  va_list ap;
  settings.ensureInit();
  if (argc < 0 || argc > kMaxArgs)
    fatal("too many arguments to an outlined function");
  va_start(ap, fn);
  for (int32_t i = 0; i < argc; ++i)
    argv[i] = va_arg(ap, void *);
  va_end(ap);
  forkCall(fn, argc, argv);
}

void __kmpc_push_num_threads(void *, int32_t, int32_t nth) {
  settings.ensureInit();
  if (nth < 0)
    nth = 0;
  else if (nth > kMaxThreads)
    nth = kMaxThreads;
  SelfWord::set(SelfWord::current().withPending(nth));
}

void __kmpc_push_proc_bind(void *, int32_t, int32_t) {}

void __kmpc_serialized_parallel(void *, int32_t) {
  settings.ensureInit();
  SelfWord w = SelfWord::current();
  if (w.pending())
    SelfWord::set(w = w.withoutPending());
  if (w.inForeign() || root.enter(w) == Root::Entry::Foreign) {
    SelfWord::set(SelfWord::current().enterForeign());
    return;
  }
  Thread::current()->regions().pushSerialized();
}

void __kmpc_end_serialized_parallel(void *, int32_t) {
  SelfWord w = SelfWord::current();
  if (w.inForeign()) {
    SelfWord::set(w.leaveForeign());
    return;
  }
  Thread *self = Thread::current();
  if (!self)
    fatal("__kmpc_end_serialized_parallel without a matching "
          "__kmpc_serialized_parallel");
  self->regions().pop();
  if (self->slot() == 0 && self->regions().depth() == 0)
    root.leave();
}

void __kmpc_barrier(void *, int32_t) {
  Thread *self = Thread::current();
  Level *top = self ? self->regions().activeTop() : nullptr;
  if (top)
    top->team->barrier(*self, *top);
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

// The queries of the calling thread's regions need no init: before it, no
// thread is in a region nor has a descriptor (forks and serialized regions
// init first), so they answer as outside of any region.
int omp_get_thread_num(void) {
  RegionStack *regions = Thread::currentRegions();
  return regions && !SelfWord::current().inForeign() ? regions->topTid() : 0;
}

int omp_get_num_threads(void) {
  RegionStack *regions = Thread::currentRegions();
  return regions && !SelfWord::current().inForeign() ? regions->topNth() : 1;
}

// A foreign region is nested only in other foreign regions, all inactive.
int omp_get_level(void) {
  SelfWord w = SelfWord::current();
  if (w.inForeign())
    return w.foreignDepth();
  Thread *self = Thread::current();
  return self ? self->regions().depth() : 0;
}

int omp_get_active_level(void) {
  Thread *self = Thread::current();
  return self && !SelfWord::current().inForeign() ? self->regions().active()
                                                  : 0;
}

int omp_in_parallel(void) { return omp_get_active_level() > 0; }

// The thread limit is set once per process: by OMP_THREAD_LIMIT at init, or
// else by the first pomp_open_pool(limit) with a limit in [1, 64]. With a valid
// OMP_THREAD_LIMIT these two report it from the first call, before any open.
// Until a limit is set they return 1, with one warning: every region runs on
// one thread then, as pomp only uses threads the host lends it.
int omp_get_max_threads(void) {
  int active = omp_get_active_level();
  settings.ensureInit();
  int n = settings.numThreads(active);
  if (n > 0)
    return n;
  settings.warnNoLimit();
  return 1;
}

int omp_get_thread_limit(void) {
  settings.ensureInit();
  int n = settings.limit();
  if (n > 0)
    return n;
  settings.warnNoLimit();
  return 1;
}

int omp_get_max_active_levels(void) {
  settings.ensureInit();
  return settings.maxActive();
}

} // extern "C"
