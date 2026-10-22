/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pomp.c -----------------------------------===//
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
// donated by the host (slots 1..limit-1). One 64-bit word, `busy`, holds one
// bit per slot; a bit is clear iff that slot is a donated, idle worker. A
// leader recruits a team with a single CAS on `busy`, hands each worker the
// work through its descriptor, wakes it by bumping its private epoch, and
// clears the team's bits again after the join. At most two levels are active;
// deeper or empty forks run inline as a team of one.
//
// Release does not block the leader: it closes the pool, claims the idle
// workers and marks them in `released`. Spinning workers see their bit and
// leave by themselves; each leaving worker wakes up to two sleeping ones, so
// the wake-ups spread as a tree. See POMP.md for the scenarios supported.
//
//===----------------------------------------------------------------------===//

#include "pomp.h"

#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

#if defined(__s390x__) || defined(__MVS__)
#define POMP_CACHE_LINE 256
#elif defined(__aarch64__) && defined(__APPLE__)
#define POMP_CACHE_LINE 128
#else
#define POMP_CACHE_LINE 64
#endif

#define POMP_MAX_ARGS 16  /* Max shared args passed to an outlined fn. */
#define POMP_MAX_DEPTH 8  /* Max lexical nesting of parallel regions. */
#define POMP_MAX_ACTIVE 2 /* Max active (more than 1 thread) levels. */
#define POMP_DEFAULT_SPIN                                                      \
  200000                     /* Idle-worker spin iterations before sleep.      \
                              */
#define POMP_YIELD_SPIN 4096 /* Barrier/join spin iterations before yield. */

// Schedule kinds (values of libomp's enum sched_type).
#define POMP_SCH_STATIC_CHUNKED 33
#define POMP_SCH_STATIC 34
#define POMP_SCH_MODIFIER_MASK ((1 << 29) | (1 << 30))

typedef void (*pomp_microtask_t)(int32_t *gtid, int32_t *btid, ...);

// -----------------------------------------------------------------------------
// Data structures, all statically allocated
// -----------------------------------------------------------------------------

// A team descriptor: written by its leader before waking the members, read
// by the members, and owned by the leader again after the join.
typedef struct pomp_team {
  // Work, written before the wake-up of the members.
  pomp_microtask_t fn;
  int32_t argc;
  void *argv[POMP_MAX_ARGS];
  int32_t nth;   /* Team size, leader included. */
  int32_t level; /* Active level of this team: 1 or 2. */
  uint8_t slot_of_tid[POMP_MAX_THREADS];
  // Reuse cache: the last request and the workers it got.
  int32_t cached_req;
  uint64_t cached_mask;
  // For a level-1 team: preferred workers for the level-2 team of each tid.
  uint64_t sub_mask[POMP_MAX_THREADS];
  // Join: number of members (leader excluded) done with the region.
  _Alignas(POMP_CACHE_LINE) atomic_int arrived;
  // Barrier: arrival count and generation.
  _Alignas(POMP_CACHE_LINE) atomic_int bar_count;
  _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t bar_epoch;
} pomp_team_t;

// One entry of a thread's region stack.
typedef struct pomp_level {
  pomp_team_t *team; /* NULL for a serialized (team of one) region. */
  int32_t tid;
  int32_t nth;
} pomp_level_t;

// Values of pomp_thread_t::present.
#define POMP_FREE 0    /* No thread donated as this slot. */
#define POMP_DONATED 1 /* A thread is donated as this slot. */
#define POMP_LEAVING 2 /* The donated thread is returning to the host. */

// A thread descriptor, one per slot.
typedef struct pomp_thread {
  // Wake-up, on its own line: the epoch the worker waits on.
  _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t go;
  // Assignment for the next wake-up, written by the recruiter before `go`.
  _Alignas(POMP_CACHE_LINE) pomp_team_t *team;
  int32_t tid;
  // Owned by the thread itself.
  int32_t depth;
  int32_t active;      /* Number of active levels in the stack. */
  int32_t pending_nth; /* From __kmpc_push_num_threads, for the next fork. */
  pomp_level_t stack[POMP_MAX_DEPTH];
  // Sleep: a binary "posted" flag with its mutex and condition.
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool posted;
  // POMP_FREE, POMP_DONATED or POMP_LEAVING.
  atomic_int present;
} pomp_thread_t;

static pomp_thread_t pomp_threads[POMP_MAX_THREADS];
// Team led by the root at level 1, and team led by slot i at level 2.
static pomp_team_t pomp_root_team;
static pomp_team_t pomp_slot_team[POMP_MAX_THREADS];

// Bit i is clear iff slot i is a donated, idle worker.
static _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t pomp_busy = UINT64_MAX;
// Bit i is set while worker i sleeps, or is about to.
static _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t pomp_sleeping;
// Bit i is set from the release that claimed worker i until it leaves.
static _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t pomp_released;
// 1 from a release until the next open: no donation is accepted, and forks
// run on their leader alone.
static _Alignas(POMP_CACHE_LINE) atomic_int pomp_closed;
static _Alignas(POMP_CACHE_LINE) atomic_int pomp_root_active;
static atomic_int pomp_num_workers;

// Settings, fixed at init.
static atomic_int pomp_initialized; /* 0: no, 1: in progress, 2: done. */
static int pomp_limit;              /* OMP_THREAD_LIMIT, root included. */
static uint64_t pomp_limit_mask;
static int pomp_nthreads[POMP_MAX_ACTIVE]; /* Default team size per level. */
static int pomp_max_active;
static long pomp_spin;

// -----------------------------------------------------------------------------
// Thread identity: the slot of the calling thread, 0 for any non-worker.
// -----------------------------------------------------------------------------

#ifdef POMP_USE_PTHREAD_KEY
static pthread_key_t pomp_slot_key;
static pthread_once_t pomp_key_once = PTHREAD_ONCE_INIT;
static void pomp_make_key(void) { pthread_key_create(&pomp_slot_key, NULL); }
static inline int32_t pomp_self(void) {
  return (int32_t)(intptr_t)pthread_getspecific(pomp_slot_key);
}
static inline void pomp_set_self(int32_t slot) {
  pthread_once(&pomp_key_once, pomp_make_key);
  pthread_setspecific(pomp_slot_key, (void *)(intptr_t)slot);
}
#else
static __thread int32_t pomp_slot_tls = 0;
static inline int32_t pomp_self(void) { return pomp_slot_tls; }
static inline void pomp_set_self(int32_t slot) { pomp_slot_tls = slot; }
#endif

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

static void pomp_fatal(const char *msg) {
  fprintf(stderr, "pomp: fatal: %s\n", msg);
  fflush(stderr);
  abort();
}

static inline void pomp_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
  __asm__ volatile("yield" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

static inline uint64_t pomp_bit(int slot) { return (uint64_t)1 << slot; }

static inline int pomp_popcount(uint64_t x) { return __builtin_popcountll(x); }

// Keep the n lowest set bits of x.
static inline uint64_t pomp_lowest_bits(uint64_t x, int n) {
  uint64_t r = 0;
  for (; n > 0 && x; --n) {
    uint64_t low = x & (~x + 1);
    r |= low;
    x &= ~low;
  }
  return r;
}

// Parse a positive integer; return -1 if the string is not one.
static long pomp_parse_int(const char *s, const char **end) {
  char *e;
  long v;
  while (*s == ' ' || *s == '\t')
    ++s;
  if (*s < '0' || *s > '9')
    return -1;
  v = strtol(s, &e, 10);
  while (*e == ' ' || *e == '\t')
    ++e;
  if (end)
    *end = e;
  return v;
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

static int pomp_do_init(void) {
  const char *s, *end;
  long v;
  int i, n;

  s = getenv("OMP_THREAD_LIMIT");
  if (!s)
    return POMP_ERR_LIMIT;
  v = pomp_parse_int(s, &end);
  if (v < 1 || v > POMP_MAX_THREADS || *end)
    return POMP_ERR_LIMIT;
  pomp_limit = (int)v;
  pomp_limit_mask = pomp_limit == 64 ? UINT64_MAX : (pomp_bit(pomp_limit) - 1);

  // OMP_NUM_THREADS: "n" or "n1,n2"; missing entries repeat the last one.
  for (i = 0; i < POMP_MAX_ACTIVE; ++i)
    pomp_nthreads[i] = pomp_limit;
  n = 0;
  s = getenv("OMP_NUM_THREADS");
  while (s && n < POMP_MAX_ACTIVE) {
    v = pomp_parse_int(s, &end);
    if (v < 1)
      break;
    pomp_nthreads[n++] = v > pomp_limit ? pomp_limit : (int)v;
    if (*end != ',')
      break;
    s = end + 1;
  }
  for (i = n > 0 ? n : POMP_MAX_ACTIVE; i < POMP_MAX_ACTIVE; ++i)
    pomp_nthreads[i] = pomp_nthreads[i - 1];

  // Max active levels: as in libomp, 1 unless asked for, or unless
  // OMP_NUM_THREADS lists more than one level.
  pomp_max_active = n > 1 ? POMP_MAX_ACTIVE : 1;
  s = getenv("OMP_MAX_ACTIVE_LEVELS");
  if (s && (v = pomp_parse_int(s, NULL)) >= 0)
    pomp_max_active = v < 1 ? 1 : (v > POMP_MAX_ACTIVE ? POMP_MAX_ACTIVE : v);

  pomp_spin = POMP_DEFAULT_SPIN;
  s = getenv("POMP_SPIN");
  if (s && (v = pomp_parse_int(s, NULL)) >= 0)
    pomp_spin = v;

  for (i = 0; i < POMP_MAX_THREADS; ++i) {
    pthread_mutex_init(&pomp_threads[i].mutex, NULL);
    pthread_cond_init(&pomp_threads[i].cond, NULL);
  }
  return POMP_OK;
}

static int pomp_init_status = POMP_OK;

int pomp_init(void) {
  int expected = 0;
  if (atomic_load_explicit(&pomp_initialized, memory_order_acquire) == 2)
    return pomp_init_status;
  if (atomic_compare_exchange_strong(&pomp_initialized, &expected, 1)) {
    pomp_init_status = pomp_do_init();
    atomic_store_explicit(&pomp_initialized, 2, memory_order_release);
  } else {
    while (atomic_load_explicit(&pomp_initialized, memory_order_acquire) != 2)
      sched_yield();
  }
  return pomp_init_status;
}

static inline void pomp_ensure_init(void) {
  if (atomic_load_explicit(&pomp_initialized, memory_order_acquire) == 2 &&
      pomp_init_status == POMP_OK)
    return;
  if (pomp_init() != POMP_OK)
    pomp_fatal("OMP_THREAD_LIMIT must be set to an integer in [1, 64]");
}

int pomp_get_thread_limit(void) {
  return atomic_load(&pomp_initialized) == 2 ? pomp_limit : 0;
}

// -----------------------------------------------------------------------------
// Wake-up and sleep
//
// An idle worker spins on its `go` epoch and on its bit in `released`, then
// sleeps. To sleep, it sets its bit in `sleeping`, re-reads both, and waits for
// a post. A waker first publishes (bumps `go`, or sets the `released` bit),
// then reads `sleeping`, and posts only to workers whose bit it clears itself.
// All these accesses are sequentially consistent, so either the worker sees
// the new value or the waker sees the bit: no wake-up is lost. Exactly one of
// them clears the bit, so every post is consumed in the sleep it was meant for.
// Spinning workers never touch a mutex, and a waker of spinning workers makes
// one load of `sleeping` and no system call.
//
// A post only means "re-check": a waker may clear the bit of a later sleep
// than the one it meant to end. The worker then re-checks, finds nothing new,
// and sleeps again.
// -----------------------------------------------------------------------------

static void pomp_post(pomp_thread_t *t) {
  pthread_mutex_lock(&t->mutex);
  t->posted = true;
  pthread_cond_signal(&t->cond);
  pthread_mutex_unlock(&t->mutex);
}

static void pomp_wait_post(pomp_thread_t *t) {
  pthread_mutex_lock(&t->mutex);
  while (!t->posted)
    pthread_cond_wait(&t->cond, &t->mutex);
  t->posted = false;
  pthread_mutex_unlock(&t->mutex);
}

// Post to the sleepers among `mask` whose sleeping bit this call clears.
// Return the number of posts.
static int pomp_wake_sleepers(uint64_t mask) {
  uint64_t m;
  if (!mask)
    return 0;
  m = atomic_fetch_and(&pomp_sleeping, ~mask) & mask;
  for (uint64_t r = m; r; r &= r - 1)
    pomp_post(&pomp_threads[__builtin_ctzll(r)]);
  return pomp_popcount(m);
}

// Wake up to n sleeping workers that a release claimed. Used by each leaving
// worker, so that sleepers are woken as a tree, and not by the leader.
static void pomp_wake_released(int n) {
  while (n > 0) {
    uint64_t s = atomic_load(&pomp_sleeping);
    s &= atomic_load(&pomp_released);
    if (!s)
      return;
    n -= pomp_wake_sleepers(pomp_lowest_bits(s, n));
  }
}

// Wait for work or for release. Return true for work, with `*last` updated to
// the new epoch, and false when a release has claimed this worker.
static bool pomp_wait(pomp_thread_t *t, uint64_t bit, uint64_t *last) {
  uint64_t g;
  bool rel;
  long i;
  for (i = 0; i < pomp_spin; ++i) {
    g = atomic_load_explicit(&t->go, memory_order_acquire);
    if (g != *last)
      goto work;
    if (atomic_load_explicit(&pomp_released, memory_order_relaxed) & bit)
      return false;
    pomp_relax();
  }
  for (;;) {
    atomic_fetch_or(&pomp_sleeping, bit);
    g = atomic_load(&t->go);
    rel = (atomic_load(&pomp_released) & bit) != 0;
    if (g != *last || rel) {
      // A waker that cleared the bit has posted, or will: consume it.
      if (!(atomic_fetch_and(&pomp_sleeping, ~bit) & bit))
        pomp_wait_post(t);
      break;
    }
    pomp_wait_post(t);
    g = atomic_load(&t->go);
    rel = (atomic_load(&pomp_released) & bit) != 0;
    if (g != *last || rel)
      break;
  }
  if (g == *last)
    return false;
work:
  // Work and release never come together: both claim the busy bit.
  *last = g;
  return true;
}

// Spin on a condition owned by a team (join, barrier), yielding when long.
#define POMP_SPIN_UNTIL(cond)                                                  \
  do {                                                                         \
    unsigned _n = 0;                                                           \
    while (!(cond)) {                                                          \
      if (++_n < POMP_YIELD_SPIN)                                              \
        pomp_relax();                                                          \
      else                                                                     \
        sched_yield();                                                         \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// Invoking an outlined function
//
// Outlined functions are not variadic. Call each through its exact type, as
// libomp's portable kmp_invoke_microtask.cpp does, so that the call is valid
// on ABIs where variadic and fixed calls differ.
// -----------------------------------------------------------------------------

#define A(i) argv[i]
static void pomp_invoke(
    pomp_microtask_t fn, int32_t gtid, int32_t tid, int argc, void **argv) {
  typedef int32_t *I;
  typedef void *P;
  switch (argc) {
  case 0:
    ((void (*)(I, I))fn)(&gtid, &tid);
    break;
  case 1:
    ((void (*)(I, I, P))fn)(&gtid, &tid, A(0));
    break;
  case 2:
    ((void (*)(I, I, P, P))fn)(&gtid, &tid, A(0), A(1));
    break;
  case 3:
    ((void (*)(I, I, P, P, P))fn)(&gtid, &tid, A(0), A(1), A(2));
    break;
  case 4:
    ((void (*)(I, I, P, P, P, P))fn)(&gtid, &tid, A(0), A(1), A(2), A(3));
    break;
  case 5:
    ((void (*)(I, I, P, P, P, P, P))fn)(
        &gtid, &tid, A(0), A(1), A(2), A(3), A(4));
    break;
  case 6:
    ((void (*)(I, I, P, P, P, P, P, P))fn)(
        &gtid, &tid, A(0), A(1), A(2), A(3), A(4), A(5));
    break;
  case 7:
    ((void (*)(I, I, P, P, P, P, P, P, P))fn)(
        &gtid, &tid, A(0), A(1), A(2), A(3), A(4), A(5), A(6));
    break;
  case 8:
    ((void (*)(I, I, P, P, P, P, P, P, P, P))fn)(
        &gtid, &tid, A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7));
    break;
  case 9:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P))fn)(
        &gtid, &tid, A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8));
    break;
  case 10:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P))fn)(&gtid, &tid, A(0), A(1),
        A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9));
    break;
  case 11:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P, P))fn)(&gtid, &tid, A(0),
        A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10));
    break;
  case 12:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P, P, P))fn)(&gtid, &tid, A(0),
        A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10), A(11));
    break;
  case 13:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(&gtid, &tid,
        A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10),
        A(11), A(12));
    break;
  case 14:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(&gtid, &tid,
        A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10),
        A(11), A(12), A(13));
    break;
  case 15:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(&gtid,
        &tid, A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10),
        A(11), A(12), A(13), A(14));
    break;
  case 16:
    ((void (*)(I, I, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))fn)(&gtid,
        &tid, A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10),
        A(11), A(12), A(13), A(14), A(15));
    break;
  default:
    pomp_fatal("too many arguments to an outlined function");
  }
}
#undef A

// -----------------------------------------------------------------------------
// Region stack
// -----------------------------------------------------------------------------

static inline void pomp_push(
    pomp_thread_t *self, pomp_team_t *team, int32_t tid, int32_t nth) {
  if (self->depth >= POMP_MAX_DEPTH)
    pomp_fatal("parallel regions nested too deeply");
  self->stack[self->depth].team = team;
  self->stack[self->depth].tid = tid;
  self->stack[self->depth].nth = nth;
  self->depth++;
  if (nth > 1)
    self->active++;
}

static inline void pomp_pop(pomp_thread_t *self) {
  self->depth--;
  if (self->stack[self->depth].nth > 1)
    self->active--;
}

static inline pomp_level_t *pomp_top(pomp_thread_t *self) {
  return self->depth > 0 ? &self->stack[self->depth - 1] : NULL;
}

// -----------------------------------------------------------------------------
// Fork / join
// -----------------------------------------------------------------------------

// Recruit up to `want` idle workers, preferring `prefer`. Return their mask.
static uint64_t pomp_recruit(pomp_team_t *team, int want, uint64_t prefer) {
  uint64_t old = atomic_load_explicit(&pomp_busy, memory_order_relaxed);
  for (;;) {
    uint64_t pick;
    uint64_t idle = ~old & pomp_limit_mask;
    if (team->cached_req == want && team->cached_mask &&
        (team->cached_mask & ~idle) == 0) {
      pick = team->cached_mask; // Same request, same workers all idle: O(1).
    } else {
      // Take what is idle now, preferred workers first; fewer is fine.
      pick = pomp_lowest_bits(idle & prefer, want);
      pick |= pomp_lowest_bits(idle & ~pick, want - pomp_popcount(pick));
    }
    if (pick == 0)
      return 0;
    if (atomic_compare_exchange_weak_explicit(&pomp_busy, &old, old | pick,
            memory_order_acquire, memory_order_relaxed)) {
      if (pomp_popcount(pick) == want) {
        team->cached_req = want;
        team->cached_mask = pick;
      }
      return pick;
    }
  }
}

// For a level-1 team, split the remaining idle workers into one contiguous
// group per member, which that member then prefers for its level-2 team, so
// that concurrent inner forks do not compete and their caches stay valid.
static void pomp_partition(pomp_team_t *team) {
  uint64_t spare =
      ~atomic_load_explicit(&pomp_busy, memory_order_relaxed) & pomp_limit_mask;
  int n = pomp_popcount(spare), nth = team->nth, t;
  for (t = 0; t < nth; ++t) {
    int k = n / nth + (t < n % nth ? 1 : 0);
    team->sub_mask[t] = pomp_lowest_bits(spare, k);
    spare &= ~team->sub_mask[t];
  }
}

void __kmpc_fork_call(void *loc, int32_t argc, pomp_microtask_t fn, ...) {
  int32_t gtid, want, nth, tid, i;
  pomp_thread_t *self;
  pomp_level_t *top;
  pomp_team_t *team;
  void *argv[POMP_MAX_ARGS];
  uint64_t mask, prefer, m;
  va_list ap;
  bool outermost;
  (void)loc;

  pomp_ensure_init();
  if (argc < 0 || argc > POMP_MAX_ARGS)
    pomp_fatal("too many arguments to an outlined function");
  va_start(ap, fn);
  for (i = 0; i < argc; ++i)
    argv[i] = va_arg(ap, void *);
  va_end(ap);

  gtid = pomp_self();
  self = &pomp_threads[gtid];
  want = self->pending_nth;
  self->pending_nth = 0;
  if (want <= 0)
    want = pomp_nthreads[self->active < POMP_MAX_ACTIVE ? self->active
                                                        : POMP_MAX_ACTIVE - 1];
  if (want > pomp_limit)
    want = pomp_limit;

  outermost = gtid == 0 && self->depth == 0;
  if (outermost) {
    int expected = 0;
    if (!atomic_compare_exchange_strong(&pomp_root_active, &expected, 1))
      pomp_fatal("parallel regions started by two non-worker threads at once");
  }

  // Serialized: one thread asked for, no active level left, or pool closed.
  mask = 0;
  if (want > 1 && self->active < pomp_max_active &&
      !atomic_load_explicit(&pomp_closed, memory_order_acquire)) {
    if (self->active == 0) {
      if (gtid != 0)
        pomp_fatal("a worker started a parallel region outside of a team");
      team = &pomp_root_team;
      prefer = pomp_limit_mask;
    } else {
      team = &pomp_slot_team[gtid];
      top = pomp_top(self);
      prefer = top->team ? top->team->sub_mask[top->tid] : 0;
    }
    mask = pomp_recruit(team, want - 1, prefer);
  }
  if (mask == 0) {
    pomp_push(self, NULL, 0, 1);
    pomp_invoke(fn, gtid, 0, argc, argv);
    pomp_pop(self);
    if (outermost)
      atomic_store_explicit(&pomp_root_active, 0, memory_order_release);
    return;
  }

  // Describe the work.
  nth = 1 + pomp_popcount(mask);
  team->fn = fn;
  team->argc = argc;
  memcpy(team->argv, argv, sizeof(void *) * (size_t)argc);
  team->nth = nth;
  team->level = self->active + 1;
  team->slot_of_tid[0] = (uint8_t)gtid;
  atomic_store_explicit(&team->arrived, 0, memory_order_relaxed);
  atomic_store_explicit(&team->bar_count, 0, memory_order_relaxed);
  if (team->level == 1 && pomp_max_active > 1)
    pomp_partition(team);

  // Assign and wake the members; post only to those that sleep.
  for (tid = 1, m = mask; m; ++tid) {
    int w = __builtin_ctzll(m);
    pomp_thread_t *t = &pomp_threads[w];
    m &= m - 1;
    team->slot_of_tid[tid] = (uint8_t)w;
    t->team = team;
    t->tid = tid;
    atomic_fetch_add(&t->go, 1);
  }
  pomp_wake_sleepers(atomic_load(&pomp_sleeping) & mask);

  // Run as tid 0, then join.
  pomp_push(self, team, 0, nth);
  pomp_invoke(fn, gtid, 0, argc, argv);
  pomp_pop(self);
  POMP_SPIN_UNTIL(
      atomic_load_explicit(&team->arrived, memory_order_acquire) == nth - 1);
  atomic_fetch_and_explicit(&pomp_busy, ~mask, memory_order_release);
  if (outermost)
    atomic_store_explicit(&pomp_root_active, 0, memory_order_release);
}

// -----------------------------------------------------------------------------
// Host interface: donating and releasing worker threads
// -----------------------------------------------------------------------------

int pomp_donate_thread(int slot) {
  pomp_thread_t *self;
  uint64_t bit, epoch;

  if (pomp_init() != POMP_OK)
    return POMP_ERR_LIMIT;
  if (slot < 1 || slot >= pomp_limit)
    return POMP_ERR_SLOT;
  if (pomp_self() != 0)
    return POMP_ERR_STATE; // A donated thread cannot donate itself again.
  self = &pomp_threads[slot];
  bit = pomp_bit(slot);
  // Take the slot. If its previous donor was released but has not left yet,
  // wait for it: this costs this thread, not the leader. A slot whose donor
  // has not been released is in use.
  for (;;) {
    int expected = POMP_FREE;
    if (atomic_compare_exchange_weak(&self->present, &expected, POMP_DONATED))
      break;
    if (expected == POMP_DONATED && !(atomic_load(&pomp_released) & bit))
      return POMP_ERR_SLOT;
    sched_yield();
  }

  pomp_set_self(slot);
  self->depth = 0;
  self->active = 0;
  self->pending_nth = 0;
  pthread_mutex_lock(&self->mutex);
  self->posted = false;
  pthread_mutex_unlock(&self->mutex);
  epoch = atomic_load(&self->go);

  // Become idle, hence recruitable, then check the gate. Release closes the
  // gate before it claims idle workers, so either this thread sees it closed
  // or the release claims it.
  atomic_fetch_and(&pomp_busy, ~bit);
  if (atomic_load(&pomp_closed)) {
    if (!(atomic_fetch_or(&pomp_busy, bit) & bit)) {
      // Nobody claimed this slot: back out.
      pomp_set_self(0);
      atomic_store_explicit(&self->present, POMP_FREE, memory_order_release);
      return POMP_ERR_CLOSED;
    }
    // Claimed between the two steps, by a release or by a fork after an
    // open: serve as any worker.
  }
  atomic_fetch_add(&pomp_num_workers, 1);

  while (pomp_wait(self, bit, &epoch)) {
    pomp_team_t *team = self->team;
    pomp_push(self, team, self->tid, team->nth);
    pomp_invoke(team->fn, slot, self->tid, team->argc, team->argv);
    pomp_pop(self);
    // Last access to the team: after this, the leader may reuse it.
    atomic_fetch_add_explicit(&team->arrived, 1, memory_order_release);
  }

  // Released. The busy bit stays set until the slot is donated again.
  atomic_store(&self->present, POMP_LEAVING);
  atomic_fetch_and(&pomp_released, ~bit);
  pomp_wake_released(2);
  pomp_set_self(0);
  atomic_fetch_sub(&pomp_num_workers, 1);
  atomic_store_explicit(&self->present, POMP_FREE, memory_order_release);
  return POMP_OK;
}

int pomp_release_workers(void) {
  uint64_t old, idle;
  if (atomic_load(&pomp_initialized) != 2 || pomp_init_status != POMP_OK)
    return POMP_ERR_STATE;
  if (pomp_self() != 0 || pomp_threads[0].depth != 0)
    return POMP_ERR_STATE;
  // Close the gate first, so a donation from now on either backs out or is
  // claimed just below.
  atomic_store(&pomp_closed, 1);
  // Claim every idle worker, as a fork would.
  old = atomic_load(&pomp_busy);
  do {
    idle = ~old & pomp_limit_mask & ~(uint64_t)1;
  } while (idle && !atomic_compare_exchange_weak(&pomp_busy, &old, old | idle));
  if (!idle)
    return POMP_OK;
  atomic_fetch_or(&pomp_released, idle);
  // Spinning workers see their bit and leave; each leaving worker wakes
  // sleeping ones. Only if every claimed worker sleeps is there nobody to start
  // that chain: then post to one.
  if ((atomic_load(&pomp_sleeping) & idle) == idle)
    pomp_wake_sleepers(idle & (~idle + 1));
  return POMP_OK;
}

int pomp_reclaim_workers(void) {
  int i;
  if (atomic_load(&pomp_initialized) != 2 || pomp_init_status != POMP_OK)
    return POMP_ERR_STATE;
  if (pomp_self() != 0)
    return POMP_ERR_STATE;
  // Wake any claimed worker still asleep, until all have left.
  while (atomic_load(&pomp_released) & pomp_limit_mask) {
    pomp_wake_released(POMP_MAX_THREADS);
    sched_yield();
  }
  for (i = 1; i < pomp_limit; ++i)
    POMP_SPIN_UNTIL(atomic_load_explicit(&pomp_threads[i].present,
                        memory_order_acquire) != POMP_LEAVING);
  return POMP_OK;
}

int pomp_open_workers(void) {
  if (pomp_init() != POMP_OK)
    return POMP_ERR_LIMIT;
  if (pomp_self() != 0 || pomp_threads[0].depth != 0)
    return POMP_ERR_STATE;
  atomic_store(&pomp_closed, 0);
  return POMP_OK;
}

int pomp_is_open(void) { return !atomic_load(&pomp_closed); }

int pomp_num_donated(void) { return atomic_load(&pomp_num_workers); }

// -----------------------------------------------------------------------------
// Other __kmpc entry points
// -----------------------------------------------------------------------------

int32_t __kmpc_global_thread_num(void *loc) {
  (void)loc;
  return pomp_self();
}

void __kmpc_push_num_threads(void *loc, int32_t gtid, int32_t nth) {
  (void)loc;
  pomp_threads[gtid].pending_nth = nth;
}

void __kmpc_push_proc_bind(void *loc, int32_t gtid, int32_t kind) {
  (void)loc;
  (void)gtid;
  (void)kind;
}

void __kmpc_serialized_parallel(void *loc, int32_t gtid) {
  (void)loc;
  pomp_threads[gtid].pending_nth = 0;
  pomp_push(&pomp_threads[gtid], NULL, 0, 1);
}

void __kmpc_end_serialized_parallel(void *loc, int32_t gtid) {
  (void)loc;
  pomp_pop(&pomp_threads[gtid]);
}

void __kmpc_barrier(void *loc, int32_t gtid) {
  pomp_level_t *top = pomp_top(&pomp_threads[gtid]);
  pomp_team_t *team;
  uint64_t epoch;
  (void)loc;
  if (!top || top->nth == 1)
    return;
  team = top->team;
  epoch = atomic_load_explicit(&team->bar_epoch, memory_order_acquire);
  if (atomic_fetch_add_explicit(&team->bar_count, 1, memory_order_acq_rel) ==
      top->nth - 1) {
    atomic_store_explicit(&team->bar_count, 0, memory_order_relaxed);
    atomic_store_explicit(&team->bar_epoch, epoch + 1, memory_order_release);
  } else {
    POMP_SPIN_UNTIL(
        atomic_load_explicit(&team->bar_epoch, memory_order_acquire) != epoch);
  }
}

void __kmpc_for_static_fini(void *loc, int32_t gtid) {
  (void)loc;
  (void)gtid;
}

// Static loop partitioning, matching libomp's (kmp_sched.cpp) results for
// kmp_sch_static and kmp_sch_static_chunked. For kmp_sch_static, libomp's
// effective flavor is "balanced": __kmp_do_serial_initialize overrides the
// "greedy" initializer of __kmp_static.
#define POMP_STATIC_INIT(NAME, T, ST, UT)                                      \
  void NAME(void *loc, int32_t gtid, int32_t sched, int32_t *plast, T *plower, \
      T *pupper, ST *pstride, ST incr, ST chunk) {                             \
    pomp_level_t *top = pomp_top(&pomp_threads[gtid]);                         \
    UT tid, nth, trip;                                                         \
    (void)loc;                                                                 \
    sched &= ~POMP_SCH_MODIFIER_MASK;                                          \
    if (incr > 0 ? (*pupper < *plower) : (*plower < *pupper)) {                \
      if (plast)                                                               \
        *plast = 0;                                                            \
      *pstride = incr;                                                         \
      return;                                                                  \
    }                                                                          \
    nth = top ? (UT)top->nth : 1;                                              \
    tid = top ? (UT)top->tid : 0;                                              \
    if (nth == 1) {                                                            \
      if (plast)                                                               \
        *plast = 1;                                                            \
      *pstride = incr > 0 ? (ST)(*pupper - *plower + 1)                        \
                          : (ST)(-(*plower - *pupper + 1));                    \
      return;                                                                  \
    }                                                                          \
    if (incr == 1)                                                             \
      trip = (UT)(*pupper - *plower) + 1;                                      \
    else if (incr == -1)                                                       \
      trip = (UT)(*plower - *pupper) + 1;                                      \
    else if (incr > 0)                                                         \
      trip = (UT)(*pupper - *plower) / (UT)incr + 1;                           \
    else                                                                       \
      trip = (UT)(*plower - *pupper) / (UT)(-incr) + 1;                        \
    if (sched == POMP_SCH_STATIC) {                                            \
      if (trip < nth) {                                                        \
        if (tid < trip)                                                        \
          *pupper = *plower = *plower + (T)(tid * (UT)incr);                   \
        else                                                                   \
          *plower = *pupper + (incr > 0 ? 1 : -1);                             \
        if (plast)                                                             \
          *plast = (tid == trip - 1);                                          \
      } else {                                                                 \
        UT small = trip / nth, extras = trip % nth;                            \
        *plower +=                                                             \
            (T)((UT)incr * (tid * small + (tid < extras ? tid : extras)));     \
        *pupper =                                                              \
            *plower + (T)(small * (UT)incr) - (tid < extras ? 0 : (T)incr);    \
        if (plast)                                                             \
          *plast = (tid == nth - 1);                                           \
      }                                                                        \
      *pstride = (ST)trip;                                                     \
    } else if (sched == POMP_SCH_STATIC_CHUNKED) {                             \
      ST span;                                                                 \
      UT nchunks;                                                              \
      if (chunk < 1)                                                           \
        chunk = 1;                                                             \
      else if ((UT)chunk > trip)                                               \
        chunk = (ST)trip;                                                      \
      nchunks = trip / (UT)chunk + (trip % (UT)chunk ? 1 : 0);                 \
      span = chunk * incr;                                                     \
      if (nchunks < nth) {                                                     \
        *pstride = span * (ST)nchunks;                                         \
        if (tid < nchunks) {                                                   \
          *plower = *plower + (T)(span * (ST)tid);                             \
          *pupper = *plower + (T)span - (T)incr;                               \
        } else {                                                               \
          *plower = *pupper + (incr > 0 ? 1 : -1);                             \
        }                                                                      \
      } else {                                                                 \
        *pstride = span * (ST)nth;                                             \
        *plower = *plower + (T)(span * (ST)tid);                               \
        *pupper = *plower + (T)span - (T)incr;                                 \
      }                                                                        \
      if (plast)                                                               \
        *plast = (tid == (nchunks - 1) % nth);                                 \
    } else {                                                                   \
      pomp_fatal("unsupported loop schedule (only static is implemented)");    \
    }                                                                          \
  }

POMP_STATIC_INIT(__kmpc_for_static_init_4, int32_t, int32_t, uint32_t)
POMP_STATIC_INIT(__kmpc_for_static_init_4u, uint32_t, int32_t, uint32_t)
POMP_STATIC_INIT(__kmpc_for_static_init_8, int64_t, int64_t, uint64_t)
POMP_STATIC_INIT(__kmpc_for_static_init_8u, uint64_t, int64_t, uint64_t)

// -----------------------------------------------------------------------------
// omp_* queries
// -----------------------------------------------------------------------------

int omp_get_thread_num(void) {
  pomp_level_t *top = pomp_top(&pomp_threads[pomp_self()]);
  return top ? top->tid : 0;
}

int omp_get_num_threads(void) {
  pomp_level_t *top = pomp_top(&pomp_threads[pomp_self()]);
  return top ? top->nth : 1;
}

int omp_get_level(void) { return pomp_threads[pomp_self()].depth; }

int omp_get_active_level(void) { return pomp_threads[pomp_self()].active; }

int omp_in_parallel(void) { return pomp_threads[pomp_self()].active > 0; }

int omp_get_max_threads(void) {
  pomp_thread_t *self = &pomp_threads[pomp_self()];
  pomp_ensure_init();
  return pomp_nthreads[self->active < POMP_MAX_ACTIVE ? self->active
                                                      : POMP_MAX_ACTIVE - 1];
}

int omp_get_thread_limit(void) {
  pomp_ensure_init();
  return pomp_limit;
}

int omp_get_max_active_levels(void) {
  pomp_ensure_init();
  return pomp_max_active;
}
