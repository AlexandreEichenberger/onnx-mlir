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
// lent by the host through the pool (slots 1..limit-1). One 64-bit word,
// `busy`, holds one bit per slot; a bit is clear iff that slot is an idle
// worker in the pool. A
// leader recruits a team with a single CAS on `busy`, hands each worker the
// work through its descriptor, wakes it by bumping its private epoch, and
// clears the team's bits again after the join. At most two levels are active;
// deeper or empty forks run inline as a team of one.
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

// Max shared args passed to an outlined fn: one more than the 15 of libomp's
// portable kmp_invoke_microtask.cpp.
#define POMP_MAX_ARGS 16
// Max nesting levels recorded; deeper ones are serialized and only counted.
#define POMP_MAX_DEPTH 8
#define POMP_MAX_ACTIVE 2 /* Max active (more than 1 thread) levels. */
// Spin iterations of any wait (idle, join, barrier) before it sleeps.
#define POMP_DEFAULT_SPIN 200000
#define POMP_YIELD_SPIN 4096 /* Team-wait spin iterations before yield. */

// Schedule kinds (values of libomp's enum sched_type).
#define POMP_SCH_STATIC_CHUNKED 33
#define POMP_SCH_STATIC 34
#define POMP_SCH_MODIFIER_MASK ((1 << 29) | (1 << 30))

typedef void (*pomp_microtask_t)(int32_t *gtid, int32_t *btid, ...);

// -----------------------------------------------------------------------------
// Data structures, all statically allocated
// -----------------------------------------------------------------------------

// Synchronization words of one member of a team, on their own cache line.
// `done` is written by the member and polled by the leader at the join;
// `arrive` is written by the member and polled by its parent in the barrier
// tree, and `release` the other way, so a line is shared by two threads at a
// time. All three only grow, so they never need to be reset.
typedef struct pomp_sync {
  _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t done;
  atomic_uint_fast64_t arrive;
  atomic_uint_fast64_t release;
} pomp_sync_t;

// Teams of at least this many threads use the tree barrier, smaller ones the
// central one (POMP_TREE_BARRIER overrides it). Measured: on z17 the tree is
// 41-70% faster from 16 threads on and on par at 8; on Apple M-series the
// central barrier is faster up to 16 threads. See POMP.md.
#if defined(__s390x__) || defined(__MVS__)
#define POMP_DEFAULT_TREE_BARRIER 5
#else
#define POMP_DEFAULT_TREE_BARRIER (POMP_MAX_THREADS + 1) /* Never. */
#endif
#define POMP_TREE_ARITY 4

// A team descriptor: written by its leader before waking the members, read
// by the members, and owned by the leader again after the join.
typedef struct pomp_team {
  // Work, written before the wake-up of the members.
  pomp_microtask_t fn;
  int32_t argc;
  void *argv[POMP_MAX_ARGS];
  int32_t nth;      /* Team size, leader included. */
  int32_t level;    /* Active level of this team: 1 or 2. */
  int32_t depth;    /* Leader's nesting depth before this region. */
  uint64_t region;  /* Number of this region, for the members' done flags. */
  uint64_t bar_seq; /* Barrier count before this region, for the tree. */
  uint8_t slot_of_tid[POMP_MAX_THREADS];
  // Reuse cache: the last request and the workers it got.
  int32_t cached_req;
  uint64_t cached_mask;
  // For a level-1 team: preferred workers for the level-2 team of each tid.
  uint64_t sub_mask[POMP_MAX_THREADS];
  // Central barrier, for small teams: arrival count and generation.
  _Alignas(POMP_CACHE_LINE) atomic_int bar_count;
  _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t bar_epoch;
  // Bit t is set while member t sleeps in a join or barrier wait, or is about
  // to. Written only by sleepers and their wakers, so it stays shared.
  _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t sleepers;
  // Join and tree barrier: one line per member, indexed by tid.
  pomp_sync_t sync[POMP_MAX_THREADS];
} pomp_team_t;

// One entry of a thread's region stack.
typedef struct pomp_level {
  pomp_team_t *team; /* NULL for a serialized (team of one) region. */
  int32_t tid;
  int32_t nth;
  uint64_t bar; /* Barriers this thread passed in this team, all regions. */
} pomp_level_t;

// Values of pomp_thread_t::present.
#define POMP_FREE 0    /* No thread in the pool as this slot. */
#define POMP_IN_POOL 1 /* A thread is in the pool as this slot. */
#define POMP_LEAVING 2 /* The thread is returning to the host. */

// A thread descriptor, one per slot.
typedef struct pomp_thread {
  // Wake-up, on its own line: the epoch the worker waits on.
  _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t go;
  // Assignment for the next wake-up, written by the recruiter before `go`.
  _Alignas(POMP_CACHE_LINE) pomp_team_t *team;
  int32_t tid;
  // Owned by the thread itself.
  int32_t depth;  /* Nesting level; past POMP_MAX_DEPTH, only counted. */
  int32_t active; /* Number of active levels, enclosing teams included. */
  pomp_level_t stack[POMP_MAX_DEPTH];
  // Sleep: a binary "posted" flag with its mutex and condition.
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool posted;
  // POMP_FREE, POMP_IN_POOL or POMP_LEAVING.
  atomic_int present;
} pomp_thread_t;

_Static_assert(POMP_MAX_THREADS <= 64, "slots are bits of one 64-bit word");

static pomp_thread_t pomp_threads[POMP_MAX_THREADS];
// Team led by the root at level 1, and team led by slot i at level 2.
static pomp_team_t pomp_root_team;
static pomp_team_t pomp_slot_team[POMP_MAX_THREADS];

// Bit i is clear iff slot i is an idle worker in the pool.
static _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t pomp_busy = UINT64_MAX;
// Bit i is set while worker i sleeps, or is about to.
static _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t pomp_sleeping;
// Bit i is set from the release that claimed worker i until it leaves.
static _Alignas(POMP_CACHE_LINE) atomic_uint_fast64_t pomp_released;
// 1 from a close until the next open: no thread may enter, and forks
// run on their leader alone.
static _Alignas(POMP_CACHE_LINE) atomic_int pomp_closed;
// 1 while a non-worker thread, the root owner, runs regions as slot 0.
static _Alignas(POMP_CACHE_LINE) atomic_int pomp_root_active;
static atomic_int pomp_num_workers;

// Settings, fixed at init.
static atomic_int pomp_initialized; /* 0: no, 1: in progress, 2: done. */
static int pomp_env_nthreads[POMP_MAX_ACTIVE]; /* OMP_NUM_THREADS, 0: unset. */
static int pomp_max_active;
// Settings that depend on the thread limit, set once: by init from
// OMP_THREAD_LIMIT, or else by the first pomp_open_pool that gives one.
// The pool stays closed until then, and the open that sets them publishes
// them with its store to `closed`.
static pthread_mutex_t pomp_limit_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_int pomp_limit; /* Root included; 0 until set. */
static atomic_uint_fast64_t pomp_limit_mask;
static atomic_int pomp_nthreads[POMP_MAX_ACTIVE]; /* Default team sizes. */
#define POMP_LIMIT() atomic_load_explicit(&pomp_limit, memory_order_relaxed)
#define POMP_LIMIT_MASK()                                                      \
  atomic_load_explicit(&pomp_limit_mask, memory_order_relaxed)
#define POMP_NTHREADS(level)                                                   \
  atomic_load_explicit(&pomp_nthreads[level], memory_order_relaxed)
static long pomp_spin;
static int pomp_tree_min; /* Team size from which the tree barrier is used. */

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

static void pomp_fatal(const char *msg) {
  fprintf(stderr, "pomp: fatal: %s\n", msg);
  fflush(stderr);
  abort();
}

// Abort if a pthread call fails: pomp has no way to continue without it.
#define POMP_PT(call)                                                          \
  do {                                                                         \
    if ((call) != 0)                                                           \
      pomp_fatal("pthread call failed: " #call);                               \
  } while (0)

// -----------------------------------------------------------------------------
// Thread identity, one per-thread word: the slot of the calling thread (0 for
// any non-worker), plus POMP_ROOT_OWNER while a non-worker runs regions as
// slot 0. Every non-worker reads slot 0, so the flag is what tells the root of
// the running regions from another host thread.
// -----------------------------------------------------------------------------

// The word also holds what belongs to the thread rather than to its slot,
// which other non-workers share: the team size pushed by
// __kmpc_push_num_threads for its next fork, and the depth of the regions it
// runs inline while another thread owns slot 0 ("foreign" regions).
#define POMP_SLOT_BITS 0xff
#define POMP_ROOT_OWNER 0x100
#define POMP_PENDING_SHIFT 9
#define POMP_PENDING_BITS (0x7f << POMP_PENDING_SHIFT)
#define POMP_FOREIGN_ONE (1 << 16)
#define POMP_FOREIGN_BITS (0x7fff * POMP_FOREIGN_ONE)
// Set iff the thread is a worker, or inside a region.
#define POMP_IN_POMP_BITS (POMP_SLOT_BITS | POMP_ROOT_OWNER | POMP_FOREIGN_BITS)

#ifdef POMP_USE_PTHREAD_KEY
// Created by pomp_do_init. Until then no thread has a slot, and the key must
// not be read: its value would name a key of another library.
static pthread_key_t pomp_slot_key;
static atomic_int pomp_key_ready;
static inline int32_t pomp_self_word(void) {
  if (!atomic_load_explicit(&pomp_key_ready, memory_order_acquire))
    return 0;
  return (int32_t)(intptr_t)pthread_getspecific(pomp_slot_key);
}
static inline void pomp_set_self_word(int32_t w) {
  POMP_PT(pthread_setspecific(pomp_slot_key, (void *)(intptr_t)w));
}
#else
static __thread int32_t pomp_slot_tls = 0;
static inline int32_t pomp_self_word(void) { return pomp_slot_tls; }
static inline void pomp_set_self_word(int32_t w) { pomp_slot_tls = w; }
#endif

static inline int32_t pomp_self(void) {
  return pomp_self_word() & POMP_SLOT_BITS;
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

// Mask of the n lowest bits.
static inline uint64_t pomp_bits_below(int n) {
  return n >= 64 ? UINT64_MAX : pomp_bit(n) - 1;
}

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

// Set the thread limit and what depends on it, once. Return POMP_ERR_LIMIT if
// `limit` is out of range, or differs from a limit already set; 0 asks for
// the limit already set.
static atomic_int pomp_limit_mismatch_seen;

// Set the thread limit, once per process, as OpenMP's thread-limit-var: from
// OMP_THREAD_LIMIT at initialization, or else from the first open that gives
// one. Once set it never changes: a later request for another value (0 means
// none) is clamped to it, with one warning per process, and succeeds. Fails
// only while no limit is set and `limit` is not a valid one.
static int pomp_set_limit(long limit) {
  int rc = POMP_OK, cur, i;
  POMP_PT(pthread_mutex_lock(&pomp_limit_mutex));
  cur = POMP_LIMIT();
  if (cur) {
    if (limit != 0 && limit != cur &&
        !atomic_exchange(&pomp_limit_mismatch_seen, 1))
      fprintf(stderr,
          "pomp: warning: thread limit %ld requested, but the limit is "
          "already %d for this process; using %d\n",
          limit, cur, cur);
  } else if (limit < 1 || limit > POMP_MAX_THREADS) {
    rc = POMP_ERR_LIMIT;
  } else {
    for (i = 0; i < POMP_MAX_ACTIVE; ++i) {
      int n = pomp_env_nthreads[i];
      atomic_store(&pomp_nthreads[i], n && n < limit ? n : (int)limit);
    }
    atomic_store(&pomp_limit_mask, pomp_bits_below((int)limit));
    atomic_store(&pomp_limit, (int)limit);
  }
  POMP_PT(pthread_mutex_unlock(&pomp_limit_mutex));
  return rc;
}

static void pomp_do_init(void) {
  const char *s, *end;
  long v;
  int i, n, rc;

  for (i = 0; i < POMP_MAX_THREADS; ++i) {
    POMP_PT(pthread_mutex_init(&pomp_threads[i].mutex, NULL));
    POMP_PT(pthread_cond_init(&pomp_threads[i].cond, NULL));
  }
#ifdef POMP_USE_PTHREAD_KEY
  POMP_PT(pthread_key_create(&pomp_slot_key, NULL));
  atomic_store_explicit(&pomp_key_ready, 1, memory_order_release);
#endif

  // OMP_NUM_THREADS: "n" or "n1,n2"; missing entries repeat the last one.
  // Clamped to the limit once it is known.
  n = 0;
  s = getenv("OMP_NUM_THREADS");
  while (s && n < POMP_MAX_ACTIVE) {
    v = pomp_parse_int(s, &end);
    if (v < 1)
      break;
    pomp_env_nthreads[n++] = v > POMP_MAX_THREADS ? POMP_MAX_THREADS : (int)v;
    if (*end != ',')
      break;
    s = end + 1;
  }
  for (i = n > 0 ? n : POMP_MAX_ACTIVE; i < POMP_MAX_ACTIVE; ++i)
    pomp_env_nthreads[i] = pomp_env_nthreads[i - 1];

  // Max active levels: as in libomp, 1 unless asked for, or unless
  // OMP_NUM_THREADS lists more than one level.
  pomp_max_active = n > 1 ? POMP_MAX_ACTIVE : 1;
  s = getenv("OMP_MAX_ACTIVE_LEVELS");
  if (s && (v = pomp_parse_int(s, NULL)) >= 0)
    pomp_max_active = v < 1 ? 1 : (v > POMP_MAX_ACTIVE ? POMP_MAX_ACTIVE : v);

  pomp_tree_min = POMP_DEFAULT_TREE_BARRIER;
  s = getenv("POMP_TREE_BARRIER");
  if (s && (v = pomp_parse_int(s, NULL)) >= 0)
    pomp_tree_min = (int)v;

  pomp_spin = POMP_DEFAULT_SPIN;
  s = getenv("POMP_SPIN");
  if (s && (v = pomp_parse_int(s, NULL)) >= 0)
    pomp_spin = v;

  // OMP_THREAD_LIMIT, if set, must be valid. If it is not set, or not valid,
  // the pool starts closed, and the first open that gives a limit opens it.
  s = getenv("OMP_THREAD_LIMIT");
  if (s) {
    v = pomp_parse_int(s, &end);
    rc = v < 1 || *end ? POMP_ERR_LIMIT : pomp_set_limit(v);
    if (rc != POMP_OK)
      fprintf(stderr,
          "pomp: warning: OMP_THREAD_LIMIT=\"%s\" is not in [1, %d] and "
          "is ignored; the pool stays closed until pomp_open_pool "
          "gives a limit\n",
          s, POMP_MAX_THREADS);
  }
  if (!POMP_LIMIT())
    atomic_store(&pomp_closed, 1);
}

// Initialize pomp once per process, on the first call of any entry point;
// concurrent first calls wait for the one that does it. Without a valid
// thread limit, the pool stays closed and regions run on one thread: nothing
// here is fatal.
static void pomp_init(void) {
  int expected = 0;
  if (atomic_compare_exchange_strong(&pomp_initialized, &expected, 1)) {
    pomp_do_init();
    atomic_store_explicit(&pomp_initialized, 2, memory_order_release);
  } else {
    while (atomic_load_explicit(&pomp_initialized, memory_order_acquire) != 2)
      sched_yield();
  }
}

static inline void pomp_ensure_init(void) {
  if (atomic_load_explicit(&pomp_initialized, memory_order_acquire) != 2)
    pomp_init();
}

int pomp_get_thread_limit(void) {
  pomp_ensure_init();
  return POMP_LIMIT();
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
  POMP_PT(pthread_mutex_lock(&t->mutex));
  t->posted = true;
  POMP_PT(pthread_cond_signal(&t->cond));
  POMP_PT(pthread_mutex_unlock(&t->mutex));
}

static void pomp_wait_post(pomp_thread_t *t) {
  POMP_PT(pthread_mutex_lock(&t->mutex));
  while (!t->posted)
    POMP_PT(pthread_cond_wait(&t->cond, &t->mutex));
  t->posted = false;
  POMP_PT(pthread_mutex_unlock(&t->mutex));
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

// -----------------------------------------------------------------------------
// Waits inside a team (join, barriers)
//
// The same protocol as the idle wait, with the team's `sleepers` word in place
// of `sleeping`, and the bit of the waiter's tid. A thread is in at most one
// wait at a time, so it sleeps on its own slot's post, and every post is
// consumed by the wait that set the bit. A waker publishes what it waits for,
// then calls pomp_team_wake; a seq_cst fence on each side orders the publish
// against the read of `sleepers`, and the bit against the read of the
// condition.
// -----------------------------------------------------------------------------

// Wait, as member TID of TEAM running on descriptor SELF, until COND holds:
// spin, then yield, which lets an oversubscribed waitee run, then sleep after
// POMP_SPIN iterations in all.
#define POMP_TEAM_WAIT(TEAM, SELF, TID, COND)                                  \
  do {                                                                         \
    long _i = 0;                                                               \
    uint64_t _b = pomp_bit(TID);                                               \
    while (!(COND)) {                                                          \
      if (_i++ < pomp_spin) {                                                  \
        if (_i < POMP_YIELD_SPIN)                                              \
          pomp_relax();                                                        \
        else                                                                   \
          sched_yield();                                                       \
        continue;                                                              \
      }                                                                        \
      atomic_fetch_or(&(TEAM)->sleepers, _b);                                  \
      atomic_thread_fence(memory_order_seq_cst);                               \
      if (COND) {                                                              \
        /* A waker that cleared the bit has posted, or will: consume it. */    \
        if (!(atomic_fetch_and(&(TEAM)->sleepers, ~_b) & _b))                  \
          pomp_wait_post(SELF);                                                \
        break;                                                                 \
      }                                                                        \
      pomp_wait_post(SELF);                                                    \
    }                                                                          \
  } while (0)

// Wake the members in `mask` of `team` that sleep in a team wait. Return the
// sleepers whose bit this call cleared, for the caller to post to.
static inline uint64_t pomp_team_claim(pomp_team_t *team, uint64_t mask) {
  uint64_t s;
  atomic_thread_fence(memory_order_seq_cst);
  s = atomic_load_explicit(&team->sleepers, memory_order_relaxed) & mask;
  return s ? atomic_fetch_and(&team->sleepers, ~s) & s : 0;
}

static inline void pomp_team_wake(pomp_team_t *team, uint64_t mask) {
  for (uint64_t s = pomp_team_claim(team, mask); s; s &= s - 1)
    pomp_post(&pomp_threads[team->slot_of_tid[__builtin_ctzll(s)]]);
}

// Spin on a condition outside of any team (reclaim), yielding when long.
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

// Levels past POMP_MAX_DEPTH are only counted: forks there are serialized, so
// each of them is this entry. Never written, as its team has one thread.
static pomp_level_t pomp_serial_level = {NULL, 0, 1, 0};

static inline void pomp_push(
    pomp_thread_t *self, pomp_team_t *team, int32_t tid, int32_t nth) {
  if (self->depth >= POMP_MAX_DEPTH) {
    if (nth > 1)
      pomp_fatal("internal: active region past the maximum depth");
    self->depth++;
    return;
  }
  self->stack[self->depth].team = team;
  self->stack[self->depth].tid = tid;
  self->stack[self->depth].nth = nth;
  self->stack[self->depth].bar = team ? team->bar_seq : 0;
  self->depth++;
  if (nth > 1)
    self->active++;
}

static inline void pomp_pop(pomp_thread_t *self) {
  if (self->depth <= 0)
    pomp_fatal("__kmpc_end_serialized_parallel without a matching "
               "__kmpc_serialized_parallel");
  self->depth--;
  if (self->depth < POMP_MAX_DEPTH && self->stack[self->depth].nth > 1)
    self->active--;
}

static inline pomp_level_t *pomp_top(pomp_thread_t *self) {
  if (self->depth > POMP_MAX_DEPTH)
    return &pomp_serial_level;
  return self->depth > 0 ? &self->stack[self->depth - 1] : NULL;
}

// -----------------------------------------------------------------------------
// Root ownership
//
// Every thread that is not a worker in the pool reads slot 0, so two host
// threads running models at once would share slot 0's region stack and teams.
// The first of them to start a region (fork or serialized) at depth 0 owns slot
// 0 until that region ends, and marks itself in its per-thread word. Another
// non-worker that starts a region meanwhile is detected, whatever the timing:
// its region runs inline as a team of one, without touching slot 0, and a
// warning is printed once. Entry points that take a `gtid` ignore it and use
// the calling thread's own word, so a non-owner sees no enclosing region,
// which is exactly what a team of one needs.
// -----------------------------------------------------------------------------

static atomic_int pomp_second_root_seen;

static void pomp_second_root(void) {
  if (!atomic_exchange(&pomp_second_root_seen, 1))
    fprintf(stderr, "pomp: warning: two host threads ran OpenMP regions at "
                    "once; the later ones run on one thread (see POMP.md)\n");
}

// Take root ownership for the calling non-worker `w`. Return false if another
// thread owns it.
static inline bool pomp_root_try_enter(int32_t w) {
  int expected = 0;
  // Acquire: see slot 0 as the previous owner left it.
  if (!atomic_compare_exchange_strong(&pomp_root_active, &expected, 1))
    return false;
  pomp_set_self_word(w | POMP_ROOT_OWNER);
  return true;
}

static inline void pomp_root_leave(void) {
  pomp_set_self_word(pomp_self_word() & ~POMP_ROOT_OWNER);
  atomic_store_explicit(&pomp_root_active, 0, memory_order_release);
}

static inline bool pomp_in_foreign(int32_t w) {
  return (w & POMP_FOREIGN_BITS) != 0;
}

// For a region about to start on the calling thread, whose word is `w`
// (pending bits clear): return 1 if it runs on its slot as usual, 2 if it
// does after taking root ownership, and 0 if another thread owns slot 0, in
// which case the region must run as a foreign one.
static inline int pomp_enter_slot(int32_t w) {
  if ((w & POMP_SLOT_BITS) || (w & POMP_ROOT_OWNER))
    return 1;
  if (pomp_root_try_enter(w))
    return 2;
  pomp_second_root();
  return 0;
}

// Run `fn` inline as a foreign region of one thread.
static void pomp_invoke(
    pomp_microtask_t fn, int32_t gtid, int32_t tid, int argc, void **argv);
static void pomp_run_foreign(pomp_microtask_t fn, int argc, void **argv) {
  pomp_set_self_word(pomp_self_word() + POMP_FOREIGN_ONE);
  pomp_invoke(fn, 0, 0, argc, argv);
  pomp_set_self_word(pomp_self_word() - POMP_FOREIGN_ONE);
}

// The calling thread's descriptor, or NULL for a non-worker that does not own
// slot 0, which is outside of any region. Entry points that receive a `gtid`
// use this instead: a `gtid` is only what the caller says, and the thread's
// own word cannot name another thread, nor a slot out of range.
static inline pomp_thread_t *pomp_current(void) {
  int32_t w = pomp_self_word();
  if (!(w & (POMP_SLOT_BITS | POMP_ROOT_OWNER)))
    return NULL;
  return &pomp_threads[w & POMP_SLOT_BITS];
}

// The calling thread's innermost region, or NULL outside of any.
static inline pomp_level_t *pomp_current_level(void) {
  pomp_thread_t *self = pomp_current();
  return self ? pomp_top(self) : NULL;
}

// -----------------------------------------------------------------------------
// Fork / join
// -----------------------------------------------------------------------------

// Recruit up to `want` idle workers, preferring `prefer`. Return their mask.
static uint64_t pomp_recruit(pomp_team_t *team, int want, uint64_t prefer) {
  uint64_t old = atomic_load_explicit(&pomp_busy, memory_order_relaxed);
  for (;;) {
    uint64_t pick;
    uint64_t idle = ~old & POMP_LIMIT_MASK();
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
  uint64_t spare = ~atomic_load_explicit(&pomp_busy, memory_order_relaxed) &
                   POMP_LIMIT_MASK();
  int n = pomp_popcount(spare), nth = team->nth, t;
  for (t = 0; t < nth; ++t) {
    int k = n / nth + (t < n % nth ? 1 : 0);
    team->sub_mask[t] = pomp_lowest_bits(spare, k);
    spare &= ~team->sub_mask[t];
  }
}

static void pomp_release_idle(void);

// The slot of the leader of `team`.
static inline int32_t pomp_team_leader(pomp_team_t *team) {
  return team == &pomp_root_team ? 0 : (int32_t)(team - pomp_slot_team);
}

void __kmpc_fork_call(void *loc, int32_t argc, pomp_microtask_t fn, ...) {
  int32_t gtid, want, nth, tid, i, limit, w;
  pomp_thread_t *self;
  pomp_level_t *top;
  pomp_team_t *team;
  void *argv[POMP_MAX_ARGS];
  uint64_t mask, prefer, m, bar;
  va_list ap;
  bool outermost = false;
  (void)loc;

  pomp_ensure_init();
  if (argc < 0 || argc > POMP_MAX_ARGS)
    pomp_fatal("too many arguments to an outlined function");
  va_start(ap, fn);
  for (i = 0; i < argc; ++i)
    argv[i] = va_arg(ap, void *);
  va_end(ap);

  w = pomp_self_word();
  gtid = w & POMP_SLOT_BITS;
  want = (w & POMP_PENDING_BITS) >> POMP_PENDING_SHIFT;
  if (want)
    pomp_set_self_word(w &= ~POMP_PENDING_BITS);
  if (pomp_in_foreign(w)) {
    pomp_run_foreign(fn, argc, argv);
    return;
  }
  switch (pomp_enter_slot(w)) {
  case 0:
    pomp_run_foreign(fn, argc, argv);
    return;
  case 2:
    outermost = true;
    break;
  }
  self = &pomp_threads[gtid];
  limit = POMP_LIMIT();
  if (want <= 0)
    want = POMP_NTHREADS(
        self->active < POMP_MAX_ACTIVE ? self->active : POMP_MAX_ACTIVE - 1);
  if (want > limit)
    want = limit;

  // Serialized: one thread asked for, no active level left, nested too
  // deeply, or pool closed.
  mask = 0;
  if (want > 1 && self->active < pomp_max_active &&
      self->depth < POMP_MAX_DEPTH &&
      !atomic_load_explicit(&pomp_closed, memory_order_acquire)) {
    if (self->active == 0) {
      if (gtid != 0)
        pomp_fatal("a worker started a parallel region outside of a team");
      team = &pomp_root_team;
      prefer = POMP_LIMIT_MASK();
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
      pomp_root_leave();
    return;
  }

  // Describe the work. Members start at the leader's depth and level.
  nth = 1 + pomp_popcount(mask);
  team->fn = fn;
  team->argc = argc;
  memcpy(team->argv, argv, sizeof(void *) * (size_t)argc);
  team->nth = nth;
  team->level = self->active + 1;
  team->depth = self->depth;
  team->region++;
  team->slot_of_tid[0] = (uint8_t)gtid;
  atomic_store_explicit(&team->bar_count, 0, memory_order_relaxed);
  if (team->level == 1 && pomp_max_active > 1)
    pomp_partition(team);

  // Assign and wake the members; post only to those that sleep. The whole
  // map is written first: members use it to wake each other.
  for (tid = 1, m = mask; m; ++tid, m &= m - 1)
    team->slot_of_tid[tid] = (uint8_t)__builtin_ctzll(m);
  for (tid = 1; tid < nth; ++tid) {
    pomp_thread_t *t = &pomp_threads[team->slot_of_tid[tid]];
    t->team = team;
    t->tid = tid;
    atomic_fetch_add(&t->go, 1);
  }
  pomp_wake_sleepers(atomic_load(&pomp_sleeping) & mask);

  // Run as tid 0, then join: poll each member's done flag. Plain loads of
  // distinct lines overlap, where increments of one shared counter serialize.
  // A long wait sleeps; each member wakes the leader after its done flag.
  pomp_push(self, team, 0, nth);
  pomp_invoke(fn, gtid, 0, argc, argv);
  bar = pomp_top(self)->bar;
  pomp_pop(self);
  for (tid = 1; tid < nth; ++tid)
    POMP_TEAM_WAIT(team, self, 0,
        atomic_load_explicit(&team->sync[tid].done, memory_order_acquire) ==
            team->region);
  // Every member passed the same barriers: the leader's count is the team's
  // for the next region. Written after the join, as members read it on entry.
  team->bar_seq = bar;
  // Return the workers. A release that ran meanwhile, from another host
  // thread, did not see them idle; they are released here instead. Both
  // sides are seq_cst, so one of them sees the other.
  atomic_fetch_and(&pomp_busy, ~mask);
  if (atomic_load(&pomp_closed))
    pomp_release_idle();
  if (outermost)
    pomp_root_leave();
}

// -----------------------------------------------------------------------------
// Host interface: the pool
// -----------------------------------------------------------------------------

// A team member's last access to its team, at the join: wake the leader if it
// sleeps. The leader is found from the team's address, as the team may
// already be reused.
static inline void pomp_join_wake(pomp_team_t *team) {
  if (pomp_team_claim(team, 1))
    pomp_post(&pomp_threads[pomp_team_leader(team)]);
}

int pomp_enter_pool(int slot) {
  pomp_thread_t *self;
  uint64_t bit, epoch;
  int limit;

  pomp_ensure_init();
  limit = POMP_LIMIT();
  if (!limit)
    return POMP_ERR_CLOSED; // Closed until an open gives the limit.
  if (slot < 1 || slot >= limit)
    return POMP_ERR_SLOT;
  // A thread in the pool cannot enter it again, and the leader cannot enter
  // it from inside its regions.
  if ((pomp_self_word() & POMP_IN_POMP_BITS) != 0)
    return POMP_ERR_STATE;
  self = &pomp_threads[slot];
  bit = pomp_bit(slot);
  // Take the slot. If its previous thread was released but has not left yet,
  // wait for it: this costs this thread, not the leader. A slot whose thread
  // has not been released is in use.
  for (;;) {
    int expected = POMP_FREE;
    if (atomic_compare_exchange_weak(&self->present, &expected, POMP_IN_POOL))
      break;
    if (expected == POMP_IN_POOL && !(atomic_load(&pomp_released) & bit))
      return POMP_ERR_SLOT;
    sched_yield();
  }

  pomp_set_self_word(slot);
  self->depth = 0;
  self->active = 0;
  POMP_PT(pthread_mutex_lock(&self->mutex));
  self->posted = false;
  POMP_PT(pthread_mutex_unlock(&self->mutex));
  epoch = atomic_load(&self->go);

  // Become idle, hence recruitable, then check the gate. Release closes the
  // gate before it claims idle workers, so either this thread sees it closed
  // or the release claims it.
  atomic_fetch_and(&pomp_busy, ~bit);
  if (atomic_load(&pomp_closed)) {
    if (!(atomic_fetch_or(&pomp_busy, bit) & bit)) {
      // Nobody claimed this slot: back out.
      pomp_set_self_word(0);
      atomic_store_explicit(&self->present, POMP_FREE, memory_order_release);
      return POMP_ERR_CLOSED;
    }
    // Claimed between the two steps, by a release or by a fork after an
    // open: serve as any worker.
  }
  atomic_fetch_add(&pomp_num_workers, 1);

  while (pomp_wait(self, bit, &epoch)) {
    pomp_team_t *team = self->team;
    int32_t tid = self->tid;
    uint64_t region = team->region;
    // Continue the leader's nesting: same depth, same active levels.
    self->depth = team->depth;
    self->active = team->level - 1;
    pomp_push(self, team, tid, team->nth);
    pomp_invoke(team->fn, slot, tid, team->argc, team->argv);
    pomp_pop(self);
    // After this store, the leader may reuse the team; only the wake-up of
    // the leader follows, which allows it.
    atomic_store_explicit(&team->sync[tid].done, region, memory_order_release);
    pomp_join_wake(team);
  }

  // Released. The busy bit stays set until a thread enters as this slot.
  self->depth = 0;
  self->active = 0;
  atomic_store(&self->present, POMP_LEAVING);
  atomic_fetch_and(&pomp_released, ~bit);
  pomp_wake_released(2);
  pomp_set_self_word(0);
  atomic_fetch_sub(&pomp_num_workers, 1);
  atomic_store_explicit(&self->present, POMP_FREE, memory_order_release);
  return POMP_OK;
}

// Claim every idle worker for a release, as a fork would, and start their
// wake-up. Called by a release after it closed the gate, and by a leader that
// returns its team's workers while the gate is closed.
static void pomp_release_idle(void) {
  uint64_t old = atomic_load(&pomp_busy), idle;
  do {
    idle = ~old & POMP_LIMIT_MASK() & ~(uint64_t)1;
  } while (idle && !atomic_compare_exchange_weak(&pomp_busy, &old, old | idle));
  if (!idle)
    return;
  atomic_fetch_or(&pomp_released, idle);
  // Spinning workers see their bit and leave; each leaving worker wakes
  // sleeping ones. Only if every claimed worker sleeps is there nobody to start
  // that chain: then post to one.
  if ((atomic_load(&pomp_sleeping) & idle) == idle)
    pomp_wake_sleepers(idle & (~idle + 1));
}

int pomp_close_pool(void) {
  pomp_ensure_init();
  // Not from a worker, nor from inside the regions of the calling thread.
  if ((pomp_self_word() & POMP_IN_POMP_BITS) != 0)
    return POMP_ERR_STATE;
  // Close the gate first, so an entry from now on either backs out or is
  // claimed just below.
  atomic_store(&pomp_closed, 1);
  pomp_release_idle();
  return POMP_OK;
}

int pomp_wait_pool_empty(void) {
  int i, limit;
  pomp_ensure_init();
  if ((pomp_self_word() & POMP_IN_POMP_BITS) != 0)
    return POMP_ERR_STATE;
  limit = POMP_LIMIT();
  // Wake any claimed worker still asleep, until all have left.
  while (atomic_load(&pomp_released) & POMP_LIMIT_MASK()) {
    pomp_wake_released(POMP_MAX_THREADS);
    sched_yield();
  }
  for (i = 1; i < limit; ++i)
    POMP_SPIN_UNTIL(atomic_load_explicit(&pomp_threads[i].present,
                        memory_order_acquire) != POMP_LEAVING);
  return POMP_OK;
}

int pomp_open_pool(int thread_limit) {
  int rc;
  pomp_ensure_init();
  if ((pomp_self_word() & POMP_IN_POMP_BITS) != 0)
    return POMP_ERR_STATE;
  // Whether the pool is open or closed, the outcome is the same: open with
  // the process's limit, or POMP_ERR_LIMIT with the pool unchanged.
  rc = pomp_set_limit(thread_limit);
  if (rc != POMP_OK)
    return rc;
  atomic_store(&pomp_closed, 0);
  return POMP_OK;
}

int pomp_is_pool_open(void) { return !atomic_load(&pomp_closed); }

int pomp_pool_size(void) { return atomic_load(&pomp_num_workers); }

// -----------------------------------------------------------------------------
// Other __kmpc entry points
// -----------------------------------------------------------------------------

int32_t __kmpc_global_thread_num(void *loc) {
  (void)loc;
  return pomp_self();
}

void __kmpc_push_num_threads(void *loc, int32_t gtid, int32_t nth) {
  int32_t w;
  (void)loc;
  pomp_ensure_init();
  w = pomp_self_word();
  (void)gtid;
  if (nth < 0)
    nth = 0;
  else if (nth > POMP_MAX_THREADS)
    nth = POMP_MAX_THREADS;
  pomp_set_self_word(
      (w & ~POMP_PENDING_BITS) | ((int32_t)nth << POMP_PENDING_SHIFT));
}

void __kmpc_push_proc_bind(void *loc, int32_t gtid, int32_t kind) {
  (void)loc;
  (void)gtid;
  (void)kind;
}

void __kmpc_serialized_parallel(void *loc, int32_t gtid) {
  int32_t w;
  (void)loc;
  pomp_ensure_init();
  w = pomp_self_word();
  (void)gtid;
  if (w & POMP_PENDING_BITS)
    pomp_set_self_word(w &= ~POMP_PENDING_BITS);
  if (pomp_in_foreign(w) || !pomp_enter_slot(w)) {
    pomp_set_self_word(pomp_self_word() + POMP_FOREIGN_ONE);
    return;
  }
  pomp_push(pomp_current(), NULL, 0, 1);
}

void __kmpc_end_serialized_parallel(void *loc, int32_t gtid) {
  int32_t w = pomp_self_word();
  pomp_thread_t *self;
  (void)loc;
  (void)gtid;
  if (pomp_in_foreign(w)) {
    pomp_set_self_word(w - POMP_FOREIGN_ONE);
    return;
  }
  self = pomp_current();
  if (!self)
    pomp_fatal("__kmpc_end_serialized_parallel without a matching "
               "__kmpc_serialized_parallel");
  pomp_pop(self);
  if (self == &pomp_threads[0] && self->depth == 0)
    pomp_root_leave();
}

// Tree barrier: tid t is the parent of tids ARITY*t+1 .. ARITY*t+ARITY. Each
// thread waits for its children to arrive, reports its own arrival to its
// parent and waits for the parent's release, then releases its children.
// Every flag is written by one thread and read by one other, so no line is
// contended; the depth is log_ARITY(nth).
static void pomp_tree_barrier(pomp_team_t *team, pomp_thread_t *self,
    int32_t tid, int32_t nth, uint64_t k) {
  int32_t c, first = POMP_TREE_ARITY * tid + 1;
  int32_t last = first + POMP_TREE_ARITY;
  uint64_t children = 0;
  if (last > nth)
    last = nth;
  for (c = first; c < last; ++c) {
    POMP_TEAM_WAIT(team, self, tid,
        atomic_load_explicit(&team->sync[c].arrive, memory_order_acquire) >= k);
    children |= pomp_bit(c);
  }
  if (tid != 0) {
    atomic_store_explicit(&team->sync[tid].arrive, k, memory_order_release);
    pomp_team_wake(team, pomp_bit((tid - 1) / POMP_TREE_ARITY));
    POMP_TEAM_WAIT(team, self, tid,
        atomic_load_explicit(&team->sync[tid].release, memory_order_acquire) >=
            k);
  }
  for (c = first; c < last; ++c)
    atomic_store_explicit(&team->sync[c].release, k, memory_order_release);
  if (children)
    pomp_team_wake(team, children);
}

void __kmpc_barrier(void *loc, int32_t gtid) {
  pomp_thread_t *self = pomp_current();
  pomp_level_t *top = self ? pomp_top(self) : NULL;
  pomp_team_t *team;
  uint64_t epoch;
  (void)loc;
  (void)gtid;
  if (!top || top->nth == 1)
    return;
  team = top->team;
  top->bar++;
  if (top->nth >= pomp_tree_min) {
    pomp_tree_barrier(team, self, top->tid, top->nth, top->bar);
    return;
  }
  epoch = atomic_load_explicit(&team->bar_epoch, memory_order_acquire);
  if (atomic_fetch_add_explicit(&team->bar_count, 1, memory_order_acq_rel) ==
      top->nth - 1) {
    atomic_store_explicit(&team->bar_count, 0, memory_order_relaxed);
    atomic_store_explicit(&team->bar_epoch, epoch + 1, memory_order_release);
    pomp_team_wake(team, pomp_bits_below(top->nth) & ~pomp_bit(top->tid));
  } else {
    POMP_TEAM_WAIT(team, self, top->tid,
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
//
// All arithmetic is in the unsigned type UT, where it wraps, so the results
// are libomp's bit for bit without signed overflow. Two cases differ, both at
// the ends of the IV type, which compiled code does not reach as it
// normalizes its loops to [0, trip count - 1]:
// - A loop over all values of the type has a trip count that wraps to 0.
//   libomp then divides by zero (chunked) or splits nothing; pomp runs the
//   whole loop on tid 0.
// - The empty share of a thread is libomp's upper bound plus one step, unless
//   that wraps around the end of the type, which would make it the whole
//   range; then the upper bound moves one step back instead.

// Give the calling thread an empty share of the loop that ends at `hi`.
#define POMP_EMPTY_SHARE(T, plower, pupper, hi, incr)                          \
  do {                                                                         \
    T _past = (T)((incr) > 0 ? (hi) + 1 : (hi)-1);                             \
    if ((incr) > 0 ? _past > *(pupper) : _past < *(pupper)) {                  \
      *(plower) = _past;                                                       \
    } else {                                                                   \
      *(plower) = *(pupper);                                                   \
      *(pupper) = (T)((incr) > 0 ? (hi)-1 : (hi) + 1);                         \
    }                                                                          \
  } while (0)

#define POMP_STATIC_INIT(NAME, T, ST, UT)                                      \
  void NAME(void *loc, int32_t gtid, int32_t sched, int32_t *plast, T *plower, \
      T *pupper, ST *pstride, ST incr, ST chunk) {                             \
    pomp_level_t *top = pomp_current_level();                                  \
    UT tid, nth, trip;                                                         \
    UT lo = (UT) * plower, hi = (UT) * pupper, inc = (UT)incr;                 \
    (void)loc;                                                                 \
    (void)gtid;                                                                \
    if (incr == 0)                                                             \
      pomp_fatal("loop increment of 0");                                       \
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
      *pstride = (ST)(incr > 0 ? hi - lo + 1 : hi - lo - 1);                   \
      return;                                                                  \
    }                                                                          \
    if (incr == 1)                                                             \
      trip = hi - lo + 1;                                                      \
    else if (incr == -1)                                                       \
      trip = lo - hi + 1;                                                      \
    else if (incr > 0)                                                         \
      trip = (hi - lo) / inc + 1;                                              \
    else                                                                       \
      trip = (lo - hi) / ((UT)0 - inc) + 1;                                    \
    if (sched != POMP_SCH_STATIC && sched != POMP_SCH_STATIC_CHUNKED)          \
      pomp_fatal("unsupported loop schedule (only static is implemented)");    \
    if (trip == 0) { /* All values of the type: the whole loop on tid 0. */    \
      if (tid != 0)                                                            \
        POMP_EMPTY_SHARE(T, plower, pupper, hi, incr);                         \
      if (plast)                                                               \
        *plast = tid == 0;                                                     \
      *pstride = (ST)(hi - lo);                                                \
      return;                                                                  \
    }                                                                          \
    if (sched == POMP_SCH_STATIC) {                                            \
      if (trip < nth) {                                                        \
        if (tid < trip)                                                        \
          *pupper = *plower = (T)(lo + tid * inc);                             \
        else                                                                   \
          POMP_EMPTY_SHARE(T, plower, pupper, hi, incr);                       \
        if (plast)                                                             \
          *plast = (tid == trip - 1);                                          \
      } else {                                                                 \
        UT small = trip / nth, extras = trip % nth;                            \
        lo += inc * (tid * small + (tid < extras ? tid : extras));             \
        *plower = (T)lo;                                                       \
        *pupper = (T)(lo + small * inc - (tid < extras ? 0 : inc));            \
        if (plast)                                                             \
          *plast = (tid == nth - 1);                                           \
      }                                                                        \
      *pstride = (ST)trip;                                                     \
    } else {                                                                   \
      UT c = chunk < 1 ? 1 : (UT)chunk, span, nchunks;                         \
      if (c > trip)                                                            \
        c = trip;                                                              \
      nchunks = trip / c + (trip % c ? 1 : 0);                                 \
      span = c * inc;                                                          \
      if (nchunks < nth) {                                                     \
        *pstride = (ST)(span * nchunks);                                       \
        if (tid < nchunks) {                                                   \
          lo += span * tid;                                                    \
          *plower = (T)lo;                                                     \
          *pupper = (T)(lo + span - inc);                                      \
        } else {                                                               \
          POMP_EMPTY_SHARE(T, plower, pupper, hi, incr);                       \
        }                                                                      \
      } else {                                                                 \
        *pstride = (ST)(span * nth);                                           \
        lo += span * tid;                                                      \
        *plower = (T)lo;                                                       \
        *pupper = (T)(lo + span - inc);                                        \
      }                                                                        \
      if (plast)                                                               \
        *plast = (tid == (nchunks - 1) % nth);                                 \
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
  pomp_thread_t *self = pomp_current();
  pomp_level_t *top = self ? pomp_top(self) : NULL;
  return top && !pomp_in_foreign(pomp_self_word()) ? top->tid : 0;
}

int omp_get_num_threads(void) {
  pomp_thread_t *self = pomp_current();
  pomp_level_t *top = self ? pomp_top(self) : NULL;
  return top && !pomp_in_foreign(pomp_self_word()) ? top->nth : 1;
}

// A foreign region is nested only in other foreign regions, all inactive.
int omp_get_level(void) {
  int32_t w = pomp_self_word();
  pomp_thread_t *self = pomp_current();
  if (pomp_in_foreign(w))
    return (w & POMP_FOREIGN_BITS) / POMP_FOREIGN_ONE;
  return self ? self->depth : 0;
}

int omp_get_active_level(void) {
  pomp_thread_t *self = pomp_current();
  return self && !pomp_in_foreign(pomp_self_word()) ? self->active : 0;
}

int omp_in_parallel(void) { return omp_get_active_level() > 0; }

int omp_get_max_threads(void) {
  int active = omp_get_active_level(), n;
  pomp_ensure_init();
  n = POMP_NTHREADS(active < POMP_MAX_ACTIVE ? active : POMP_MAX_ACTIVE - 1);
  return n > 0 ? n : 1;
}

int omp_get_thread_limit(void) {
  int n;
  pomp_ensure_init();
  n = POMP_LIMIT();
  return n > 0 ? n : 1;
}

int omp_get_max_active_levels(void) {
  pomp_ensure_init();
  return pomp_max_active;
}
