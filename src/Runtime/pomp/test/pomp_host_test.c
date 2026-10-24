/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ pomp_host_test.c ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Tests of the pomp host protocol: open, enter, run, close, wait, as a
// Java host would drive it. Each scenario of POMP.md "Host protocol" has a
// test here.
//
// Usage: pomp_host_test <test> [cycles]
//   fresh    : S1, new threads each cycle, joined after release
//   reuse    : S2, the same threads enter again each cycle
//   server   : S3, enter once, many inferences, close once
//   late     : S4, entries racing with close, and after it
//   swap     : S5, threads change slots and roles between cycles
//   overlap  : S6, next cycle opened while the last one still leaves
//   sleepers : S7, release while every worker sleeps (POMP_SPIN=0)
//   errors   : S8, misuse returns errors
//   tworoots : S9, two host threads run regions at once
//   nest     : S10, nesting levels seen by members of inner teams
//   sleepjoin: S11, join and barriers that sleep (POMP_SPIN=0)
//   steal    : S13, level-2 teams of varying sizes take each other's workers
//   nolimit  : S12, no OMP_THREAD_LIMIT; the limit comes from open
//   lateopen : S14, no OMP_THREAD_LIMIT; regions of host threads span the
//              open that sets the limit (and allocates the descriptors)
//   rejoin   : S15, workers leave and enter again as the same slots, many
//              times, while they sleep and are posted to (POMP_SPIN=0)
//   latency  : leader time in release, and to the next region
//
//===----------------------------------------------------------------------===//

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../pomp.h"

int omp_get_thread_num(void);
int omp_get_num_threads(void);
int omp_get_level(void);
int omp_get_active_level(void);
int32_t __kmpc_global_thread_num(void *loc);
void __kmpc_push_num_threads(void *loc, int32_t gtid, int32_t nth);
void __kmpc_barrier(void *loc, int32_t gtid);
void __kmpc_serialized_parallel(void *loc, int32_t gtid);
void __kmpc_end_serialized_parallel(void *loc, int32_t gtid);

static int limit;
static long errors;

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c);    \
      ++errors;                                                                \
    }                                                                          \
  } while (0)

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// -----------------------------------------------------------------------------
// A parallel region, as compiled code would run it: each member adds its tid.
// -----------------------------------------------------------------------------

static atomic_long region_sum;
static atomic_int region_nth;

static void region_body(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  atomic_fetch_add(&region_sum, *tid);
  if (*tid == 0)
    atomic_store(&region_nth, omp_get_num_threads());
}

void __kmpc_fork_call(void *loc, int32_t argc, void *fn, ...);

// Run one region; check that every member ran once. Return the team size.
static int run_region(void) {
  int nth;
  atomic_store(&region_sum, 0);
  __kmpc_fork_call(NULL, 0, (void *)region_body);
  nth = atomic_load(&region_nth);
  CHECK(nth >= 1 && nth <= limit);
  CHECK(atomic_load(&region_sum) == (long)nth * (nth - 1) / 2);
  return nth;
}

static void wait_pool_size(int n) {
  while (pomp_pool_size() < n)
    sched_yield();
}

// -----------------------------------------------------------------------------
// Pool threads
// -----------------------------------------------------------------------------

// A thread that enters the pool once, then exits: the "fresh threads" host.
static void *enter_once(void *arg) {
  int rc = pomp_enter_pool((int)(intptr_t)arg);
  CHECK(rc == POMP_OK || rc == POMP_ERR_CLOSED);
  return (void *)(intptr_t)rc;
}

// A persistent host thread: enters the pool whenever told to, until told to
// stop. This is the "thread pool reused by the host" pattern.
typedef struct {
  pthread_t th;
  int slot;
  atomic_int go;   /* Host's command counter. */
  atomic_int done; /* Entries completed. */
  atomic_int stop;
  atomic_int last_rc;
} host_thread_t;

static host_thread_t host[64];

static void *host_loop(void *arg) {
  host_thread_t *h = arg;
  int seen = 0;
  for (;;) {
    while (atomic_load(&h->go) == seen && !atomic_load(&h->stop))
      sched_yield();
    if (atomic_load(&h->stop))
      return NULL;
    seen = atomic_load(&h->go);
    atomic_store(&h->last_rc, pomp_enter_pool(h->slot));
    atomic_fetch_add(&h->done, 1);
  }
}

static void host_start(void) {
  for (int i = 1; i < limit; ++i) {
    memset(&host[i], 0, sizeof(host[i]));
    host[i].slot = i;
    pthread_create(&host[i].th, NULL, host_loop, &host[i]);
  }
}

static void host_enter_all(void) {
  for (int i = 1; i < limit; ++i)
    atomic_fetch_add(&host[i].go, 1);
}

static void host_stop(void) {
  for (int i = 1; i < limit; ++i) {
    atomic_store(&host[i].stop, 1);
    pthread_join(host[i].th, NULL);
  }
}

// -----------------------------------------------------------------------------
// S1: fresh threads per cycle; release, then join (threads die after use).
// -----------------------------------------------------------------------------

static void test_fresh(long cycles) {
  pthread_t th[64];
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_pool(0) == POMP_OK);
    for (int i = 1; i < limit; ++i)
      pthread_create(&th[i], NULL, enter_once, (void *)(intptr_t)i);
    wait_pool_size(limit - 1);
    for (int r = 0; r < 5; ++r)
      CHECK(run_region() == limit);
    CHECK(pomp_close_pool() == POMP_OK);
    // Joining is safe right away: close ends every entry.
    for (int i = 1; i < limit; ++i)
      pthread_join(th[i], NULL);
    CHECK(pomp_pool_size() == 0);
  }
  printf("fresh: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S2: the same host threads enter again every cycle.
// -----------------------------------------------------------------------------

static void test_reuse(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_pool(0) == POMP_OK);
    host_enter_all();
    wait_pool_size(limit - 1);
    for (int r = 0; r < 5; ++r)
      CHECK(run_region() == limit);
    CHECK(pomp_close_pool() == POMP_OK);
    // The host waits for its own threads (not the leader): reclaim.
    CHECK(pomp_wait_pool_empty() == POMP_OK);
    for (int i = 1; i < limit; ++i) {
      while (atomic_load(&host[i].done) != c + 1)
        sched_yield();
      CHECK(atomic_load(&host[i].last_rc) == POMP_OK);
    }
  }
  host_stop();
  printf("reuse: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S3: server mode: enter once, many inferences, close once.
// -----------------------------------------------------------------------------

static void test_server(long inferences) {
  pthread_t th[64];
  CHECK(pomp_open_pool(0) == POMP_OK);
  for (int i = 1; i < limit; ++i)
    pthread_create(&th[i], NULL, enter_once, (void *)(intptr_t)i);
  wait_pool_size(limit - 1);
  for (long k = 0; k < inferences; ++k)
    for (int r = 0; r < 3; ++r)
      CHECK(run_region() == limit);
  CHECK(pomp_close_pool() == POMP_OK);
  for (int i = 1; i < limit; ++i)
    pthread_join(th[i], NULL);
  printf("server: %ld inferences, errors %ld\n", inferences, errors);
}

// -----------------------------------------------------------------------------
// S4: entries racing with close, and arriving after it.
// -----------------------------------------------------------------------------

static void test_late(long cycles) {
  pthread_t th[64];
  long refused = 0, served = 0;
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_pool(0) == POMP_OK);
    // Start members and close without waiting for them: each entry
    // either is claimed by the release or refused; none may stay.
    for (int i = 1; i < limit; ++i)
      pthread_create(&th[i], NULL, enter_once, (void *)(intptr_t)i);
    if (c & 1)
      run_region(); // Uses whoever has arrived.
    CHECK(pomp_close_pool() == POMP_OK);
    for (int i = 1; i < limit; ++i) {
      void *rc;
      pthread_join(th[i], &rc); // Must not hang.
      if ((intptr_t)rc == POMP_ERR_CLOSED)
        ++refused;
      else
        ++served;
    }
    CHECK(pomp_pool_size() == 0);
    // An entry after the close, before the next open: refused at once.
    CHECK(pomp_enter_pool(1) == POMP_ERR_CLOSED);
    // While closed, regions run on the leader alone.
    CHECK(run_region() == 1);
  }
  printf("late: %ld cycles, served %ld, refused %ld, errors %ld\n", cycles,
      served, refused, errors);
}

// -----------------------------------------------------------------------------
// S5: threads change slots and roles between cycles.
// -----------------------------------------------------------------------------

static atomic_int leader_turn_done;

static void *leader_turn(void *arg) {
  (void)arg;
  // A thread that entered before now acts as the initial thread.
  for (int r = 0; r < 5; ++r)
    run_region();
  atomic_store(&leader_turn_done, 1);
  return NULL;
}

static void test_swap(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    // Rotate slots: host thread i enters as slot (i + c) mod (limit-1) + 1.
    for (int i = 1; i < limit; ++i)
      host[i].slot = (int)((i - 1 + c) % (limit - 1)) + 1;
    CHECK(pomp_open_pool(0) == POMP_OK);
    host_enter_all();
    wait_pool_size(limit - 1);
    CHECK(run_region() == limit);
    CHECK(pomp_close_pool() == POMP_OK);
    CHECK(pomp_wait_pool_empty() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      while (atomic_load(&host[i].done) != c + 1)
        sched_yield();
  }
  host_stop();
  // A former member as the initial thread: a fresh thread that ran
  // pomp_enter_pool before.
  {
    pthread_t t;
    pthread_create(&t, NULL, enter_once, (void *)(intptr_t)1);
    pthread_join(t, NULL); // Pool is closed: refused, returns at once.
    pthread_create(&t, NULL, leader_turn, NULL);
    pthread_join(t, NULL);
    CHECK(atomic_load(&leader_turn_done) == 1);
  }
  printf("swap: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S6: next cycle opened and entered while the last one is still leaving.
// -----------------------------------------------------------------------------

static void test_overlap(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_pool(0) == POMP_OK);
    host_enter_all();
    // Do not wait for everyone: regions take whoever is idle.
    for (int r = 0; r < 3; ++r)
      run_region();
    CHECK(pomp_close_pool() == POMP_OK);
    // No reclaim: the next cycle starts while workers still leave. Each host
    // thread enters again only after its previous entry returned.
    for (int i = 1; i < limit; ++i)
      while (atomic_load(&host[i].done) != c + 1)
        sched_yield();
  }
  CHECK(pomp_wait_pool_empty() == POMP_OK);
  host_stop();
  printf("overlap: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S7: release while every worker sleeps (run with POMP_SPIN=0).
// -----------------------------------------------------------------------------

static void test_sleepers(long cycles) {
  pthread_t th[64];
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_pool(0) == POMP_OK);
    for (int i = 1; i < limit; ++i)
      pthread_create(&th[i], NULL, enter_once, (void *)(intptr_t)i);
    wait_pool_size(limit - 1);
    run_region();
    // Give the workers time to fall asleep.
    struct timespec ts = {0, 2000000};
    nanosleep(&ts, NULL);
    CHECK(pomp_close_pool() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      pthread_join(th[i], NULL); // Must not hang: the chain wakes all.
  }
  printf("sleepers: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S8: misuse.
// -----------------------------------------------------------------------------

static atomic_int inner_rc;

static void release_inside(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  if (*tid == 0) {
    int rc = pomp_close_pool();
    // Open and reclaim are refused the same way.
    if (pomp_open_pool(0) != POMP_ERR_STATE ||
        pomp_wait_pool_empty() != POMP_ERR_STATE)
      rc = -1;
    atomic_store(&inner_rc, rc);
  }
}

static atomic_int inner_enter_rc;

static void enter_inside(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  if (*tid == 0)
    atomic_store(&inner_enter_rc, pomp_enter_pool(1));
}

static void *enter_twice(void *arg) {
  (void)arg;
  return (void *)(intptr_t)pomp_enter_pool(1);
}

static void test_errors(void) {
  pthread_t a, b;
  void *rc;
  CHECK(pomp_open_pool(0) == POMP_OK);
  CHECK(pomp_enter_pool(0) == POMP_ERR_SLOT);
  CHECK(pomp_enter_pool(limit) == POMP_ERR_SLOT);
  if (limit > 2) {
    pthread_create(&a, NULL, enter_once, (void *)(intptr_t)1);
    wait_pool_size(1);
    // Slot 1 is in use and not released.
    pthread_create(&b, NULL, enter_twice, NULL);
    pthread_join(b, &rc);
    CHECK((intptr_t)rc == POMP_ERR_SLOT);
    // Release from inside a region is refused.
    __kmpc_fork_call(NULL, 0, (void *)release_inside);
    CHECK(atomic_load(&inner_rc) == POMP_ERR_STATE);
    // So is an entry, and an open.
    __kmpc_fork_call(NULL, 0, (void *)enter_inside);
    CHECK(atomic_load(&inner_enter_rc) == POMP_ERR_STATE);
    // Once the limit is set, any other request is clamped to it, and the
    // pool opens, whether it was open or closed.
    CHECK(pomp_open_pool(limit == 1 ? 2 : limit - 1) == POMP_OK);
    CHECK(pomp_get_thread_limit() == limit && pomp_is_pool_open());
    CHECK(pomp_open_pool(POMP_MAX_THREADS + 1) == POMP_OK);
    CHECK(pomp_open_pool(-3) == POMP_OK);
    CHECK(pomp_close_pool() == POMP_OK);
    CHECK(pomp_open_pool(limit + 1) == POMP_OK);
    CHECK(pomp_get_thread_limit() == limit && pomp_is_pool_open());
    CHECK(pomp_open_pool(limit) == POMP_OK);
    CHECK(pomp_close_pool() == POMP_OK);
    pthread_join(a, NULL);
  }
  printf("errors: errors %ld\n", errors);
}

// -----------------------------------------------------------------------------
// Leader cost of release, and of the first region after it.
// -----------------------------------------------------------------------------

static void test_latency(long cycles) {
  double t_rel = 0, t_next = 0, t_rel_max = 0;
  host_start();
  for (long c = 0; c < cycles; ++c) {
    double t0, t1, t2;
    CHECK(pomp_open_pool(0) == POMP_OK);
    host_enter_all();
    wait_pool_size(limit - 1);
    run_region();
    t0 = now();
    CHECK(pomp_close_pool() == POMP_OK);
    t1 = now();
    run_region(); // Closed: leader alone, must not wait for anyone.
    t2 = now();
    t_rel += t1 - t0;
    t_next += t2 - t1;
    if (t1 - t0 > t_rel_max)
      t_rel_max = t1 - t0;
    CHECK(pomp_wait_pool_empty() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      while (atomic_load(&host[i].done) != c + 1)
        sched_yield();
  }
  host_stop();
  printf("latency: %d threads, release %.2f us (max %.2f), next region %.2f "
         "us, errors %ld\n",
      limit, t_rel / cycles * 1e6, t_rel_max * 1e6, t_next / cycles * 1e6,
      errors);
}

// -----------------------------------------------------------------------------
// S9: two host threads run regions at once. The later one runs its regions on
// one thread and never touches slot 0, so the owner's barriers still work.
// -----------------------------------------------------------------------------

static atomic_int owner_in, other_in, other_go, other_bad;

static void other_inner(int32_t *gtid, int32_t *tid) {
  int32_t g = __kmpc_global_thread_num(NULL);
  if (*gtid != 0 || *tid != 0 || g != 0 || omp_get_num_threads() != 1)
    atomic_fetch_add(&other_bad, 1);
  __kmpc_barrier(NULL, g); // A team of one: returns at once.
}

static void other_body(int32_t *gtid, int32_t *tid) {
  int32_t g = __kmpc_global_thread_num(NULL);
  (void)gtid;
  if (*tid != 0 || omp_get_num_threads() != 1 || omp_get_level() != 1 ||
      omp_get_active_level() != 0)
    atomic_fetch_add(&other_bad, 1);
  atomic_store(&other_in, 1);
  while (!atomic_load(&other_go))
    sched_yield();
  __kmpc_push_num_threads(NULL, g, 4);
  __kmpc_fork_call(NULL, 0, (void *)other_inner);
  __kmpc_serialized_parallel(NULL, g);
  if (omp_get_level() != 2)
    atomic_fetch_add(&other_bad, 1);
  __kmpc_end_serialized_parallel(NULL, g);
  __kmpc_barrier(NULL, g);
}

static void *other_host(void *arg) {
  (void)arg;
  while (!atomic_load(&owner_in))
    sched_yield();
  __kmpc_fork_call(NULL, 0, (void *)other_body);
  return NULL;
}

static void owner_body(int32_t *gtid, int32_t *tid) {
  if (*tid == 0) {
    atomic_store(&owner_in, 1);
    while (!atomic_load(&other_in))
      sched_yield();
  }
  // Every member must pass, whatever the other host thread does meanwhile.
  __kmpc_barrier(NULL, *gtid);
  if (*tid == 0)
    atomic_store(&other_go, 1);
  __kmpc_barrier(NULL, *gtid);
  atomic_fetch_add(&region_sum, *tid);
}

static void test_tworoots(long cycles) {
  host_start();
  CHECK(pomp_open_pool(0) == POMP_OK);
  host_enter_all();
  wait_pool_size(limit - 1);
  for (long c = 0; c < cycles; ++c) {
    pthread_t b;
    atomic_store(&owner_in, 0);
    atomic_store(&other_in, 0);
    atomic_store(&other_go, 0);
    atomic_store(&region_sum, 0);
    pthread_create(&b, NULL, other_host, NULL);
    __kmpc_fork_call(NULL, 0, (void *)owner_body);
    pthread_join(b, NULL);
    CHECK(atomic_load(&region_sum) == (long)limit * (limit - 1) / 2);
    CHECK(omp_get_level() == 0);
    // Slot 0 is free again: a region from either thread may own it.
    CHECK(run_region() == limit);
  }
  CHECK(atomic_load(&other_bad) == 0);
  CHECK(pomp_close_pool() == POMP_OK);
  host_stop();
  printf("tworoots: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S10: members of a level-2 team are at level 2, so a third parallel level is
// serialized, as in libomp. Run with OMP_MAX_ACTIVE_LEVELS=2.
// -----------------------------------------------------------------------------

static atomic_int nest_bad, nest_max_l3;

static void nest_l3(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  (void)tid;
  if (omp_get_level() != 3 || omp_get_active_level() > 2)
    atomic_fetch_add(&nest_bad, 1);
  if (omp_get_num_threads() > atomic_load(&nest_max_l3))
    atomic_store(&nest_max_l3, omp_get_num_threads());
}

static void nest_l2(int32_t *gtid, int32_t *tid) {
  int32_t g = __kmpc_global_thread_num(NULL);
  (void)gtid;
  (void)tid;
  // Members continue their leader's nesting: level 2, and active 2 when both
  // teams got workers (fewer when the limit is too small).
  if (omp_get_level() != 2 || omp_get_active_level() > 2 ||
      (omp_get_num_threads() > 1 && omp_get_active_level() != 2 && limit > 2))
    atomic_fetch_add(&nest_bad, 1);
  __kmpc_push_num_threads(NULL, g, 2);
  __kmpc_fork_call(NULL, 0, (void *)nest_l3);
}

static void nest_l1(int32_t *gtid, int32_t *tid) {
  int32_t g = __kmpc_global_thread_num(NULL);
  (void)gtid;
  (void)tid;
  __kmpc_push_num_threads(NULL, g, 2);
  __kmpc_fork_call(NULL, 0, (void *)nest_l2);
}

// Nested 12 levels: past POMP_MAX_DEPTH, levels are serialized and counted.
static atomic_int deep_level;
static void deep(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  (void)tid;
  atomic_store(&deep_level, omp_get_level());
  if (omp_get_level() < 12)
    __kmpc_fork_call(NULL, 0, (void *)deep);
}

static void test_nest(long cycles) {
  host_start();
  CHECK(pomp_open_pool(0) == POMP_OK);
  host_enter_all();
  wait_pool_size(limit - 1);
  for (long c = 0; c < cycles; ++c) {
    __kmpc_push_num_threads(NULL, 0, 2);
    __kmpc_fork_call(NULL, 0, (void *)nest_l1);
    __kmpc_fork_call(NULL, 0, (void *)deep);
    CHECK(atomic_load(&deep_level) == 12);
    CHECK(omp_get_level() == 0);
  }
  CHECK(atomic_load(&nest_bad) == 0);
  // With fewer than 4 threads, a level-2 team may get no worker. It is then
  // inactive, and its level-3 region is only the second active level, which
  // may get workers, as in libomp.
  if (limit >= 4)
    CHECK(atomic_load(&nest_max_l3) == 1);
  CHECK(pomp_close_pool() == POMP_OK);
  host_stop();
  printf("nest: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S13: with OMP_MAX_ACTIVE_LEVELS=2, level-1 and level-2 teams of sizes that
// change every cycle, so that inner teams take workers from each other's
// preferred groups and from each other's cached teams. Every member of every
// team must see the right work and a distinct tid in [0, nth).
// -----------------------------------------------------------------------------

typedef struct {
  atomic_ullong tids; /* Bit t: tid t ran. */
  atomic_int count;
  atomic_int nth;
  long cycle;
} steal_team_t;

static atomic_int steal_bad;

static void steal_l2(int32_t *gtid, int32_t *tid, steal_team_t *st) {
  int n = omp_get_num_threads();
  (void)gtid;
  if (*tid != omp_get_thread_num() || *tid >= n || omp_get_level() != 2)
    atomic_fetch_add(&steal_bad, 1);
  if (atomic_fetch_or(&st->tids, 1ull << *tid) & (1ull << *tid))
    atomic_fetch_add(&steal_bad, 1); // Two members with the same tid.
  atomic_fetch_add(&st->count, 1);
  if (*tid == 0)
    atomic_store(&st->nth, n);
}

static void steal_l1(int32_t *gtid, int32_t *tid, long *cycle) {
  int32_t g = __kmpc_global_thread_num(NULL);
  steal_team_t st = {0};
  int n1 = omp_get_num_threads(), want;
  (void)gtid;
  st.cycle = *cycle;
  // Sizes that change with the cycle and differ between members: from 1 to
  // more than this member's share, so that some teams must take workers
  // outside their group.
  want = 1 + (int)((*cycle * 7 + *tid * 3) % (limit - n1 + 2));
  __kmpc_push_num_threads(NULL, g, want);
  __kmpc_fork_call(NULL, 1, (void *)steal_l2, &st);
  {
    int n = atomic_load(&st.nth);
    if (n < 1 || n > want || atomic_load(&st.count) != n ||
        atomic_load(&st.tids) != (n >= 64 ? ~0ull : (1ull << n) - 1))
      atomic_fetch_add(&steal_bad, 1);
  }
}

static void test_steal(long cycles) {
  host_start();
  CHECK(pomp_open_pool(0) == POMP_OK);
  host_enter_all();
  wait_pool_size(limit - 1);
  for (long c = 0; c < cycles; ++c) {
    __kmpc_push_num_threads(NULL, 0, 1 + (int)(c % (limit < 4 ? limit : 4)));
    __kmpc_fork_call(NULL, 1, (void *)steal_l1, &c);
    // A plain region in between: the root's team must see the right workers.
    run_region();
  }
  CHECK(atomic_load(&steal_bad) == 0);
  CHECK(pomp_close_pool() == POMP_OK);
  host_stop();
  printf("steal: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S11: one member is slow, so the leader at the join and the others at the
// barriers sleep instead of spinning (POMP_SPIN=0 makes them sleep at once).
// -----------------------------------------------------------------------------

static atomic_long slow_cells[64];
static atomic_int slow_bad;

static void slow_body(int32_t *gtid, int32_t *tid) {
  int n = omp_get_num_threads();
  for (int b = 0; b < 3; ++b) {
    // A different member is late at each barrier, and at the join.
    if (*tid == (b + 1) % n) {
      struct timespec ts = {0, 2000000};
      nanosleep(&ts, NULL);
    }
    atomic_store(&slow_cells[*tid], b + 1);
    __kmpc_barrier(NULL, *gtid);
    for (int i = 0; i < n; ++i)
      if (atomic_load(&slow_cells[i]) < b + 1)
        atomic_fetch_add(&slow_bad, 1);
    __kmpc_barrier(NULL, *gtid);
  }
  if (*tid == n - 1) {
    struct timespec ts = {0, 2000000};
    nanosleep(&ts, NULL);
  }
  atomic_fetch_add(&region_sum, *tid);
}

static void test_sleepjoin(long cycles) {
  host_start();
  CHECK(pomp_open_pool(0) == POMP_OK);
  host_enter_all();
  wait_pool_size(limit - 1);
  for (long c = 0; c < cycles; ++c) {
    atomic_store(&region_sum, 0);
    for (int i = 0; i < 64; ++i)
      atomic_store(&slow_cells[i], 0);
    __kmpc_fork_call(NULL, 0, (void *)slow_body);
    CHECK(atomic_load(&region_sum) == (long)limit * (limit - 1) / 2);
  }
  CHECK(atomic_load(&slow_bad) == 0);
  CHECK(pomp_close_pool() == POMP_OK);
  host_stop();
  printf("sleepjoin: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S12: no OMP_THREAD_LIMIT. Until an open gives the limit, the pool is
// closed, entries are refused and regions run on one thread.
// -----------------------------------------------------------------------------

static void test_nolimit(long cycles, int lim) {
  limit = 1;
  CHECK(pomp_get_thread_limit() == 0);
  CHECK(pomp_is_pool_open() == 0);
  CHECK(pomp_enter_pool(1) == POMP_ERR_CLOSED);
  CHECK(run_region() == 1);
  // No limit yet: only a valid one opens the pool, and it is then fixed.
  CHECK(pomp_open_pool(0) == POMP_ERR_LIMIT);
  CHECK(pomp_open_pool(-1) == POMP_ERR_LIMIT);
  CHECK(pomp_open_pool(POMP_MAX_THREADS + 1) == POMP_ERR_LIMIT);
  CHECK(pomp_is_pool_open() == 0);
  CHECK(pomp_open_pool(lim) == POMP_OK);
  CHECK(pomp_get_thread_limit() == lim);
  CHECK(pomp_close_pool() == POMP_OK);
  CHECK(pomp_open_pool(lim + 1) == POMP_OK); // Clamped to lim.
  CHECK(pomp_get_thread_limit() == lim && pomp_is_pool_open());
  limit = lim;
  test_fresh(cycles);
}

// -----------------------------------------------------------------------------
// S14: no OMP_THREAD_LIMIT. Host threads run nested serialized regions, and
// regions, while another thread sets the limit with the first open. Until
// then there are no descriptors: every region runs inline, at the right level,
// on one thread. A thread inside a region when the limit is set stays inline
// until it leaves it; its next region may use the pool.
// -----------------------------------------------------------------------------

static atomic_int late_go, late_open, late_bad;

static void late_inner(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  if (*tid != 0 || omp_get_num_threads() != 1 || omp_get_level() != 2 ||
      omp_get_active_level() != 0)
    atomic_fetch_add(&late_bad, 1);
}

static void late_outer(int32_t *gtid, int32_t *tid) {
  (void)gtid;
  // Started before the open: inline, whatever happens meanwhile.
  if (*tid != 0 || omp_get_num_threads() != 1 || omp_get_level() != 1)
    atomic_fetch_add(&late_bad, 1);
  while (!atomic_load(&late_open))
    sched_yield();
  __kmpc_fork_call(NULL, 0, (void *)late_inner);
  __kmpc_serialized_parallel(NULL, 0);
  if (omp_get_level() != 2 || omp_get_num_threads() != 1)
    atomic_fetch_add(&late_bad, 1);
  __kmpc_end_serialized_parallel(NULL, 0);
}

static void *late_host(void *arg) {
  (void)arg;
  while (!atomic_load(&late_go))
    sched_yield();
  __kmpc_fork_call(NULL, 0, (void *)late_outer);
  if (omp_get_level() != 0)
    atomic_fetch_add(&late_bad, 1);
  return NULL;
}

static void test_lateopen(int lim) {
  pthread_t b;
  limit = 1;
  CHECK(pomp_get_thread_limit() == 0);
  // Before any limit: serialized and forked regions, nested, all inline.
  __kmpc_serialized_parallel(NULL, 0);
  CHECK(omp_get_level() == 1 && omp_get_num_threads() == 1);
  CHECK(run_region() == 1);
  __kmpc_end_serialized_parallel(NULL, 0);
  CHECK(omp_get_level() == 0);
  // A second host thread inside a region while this one opens.
  pthread_create(&b, NULL, late_host, NULL);
  atomic_store(&late_go, 1);
  for (int i = 0; i < 1000; ++i)
    sched_yield();
  CHECK(pomp_open_pool(lim) == POMP_OK);
  atomic_store(&late_open, 1);
  pthread_join(b, NULL);
  CHECK(atomic_load(&late_bad) == 0);
  // With the limit: the pool works, from either thread.
  limit = lim;
  host_start();
  host_enter_all();
  wait_pool_size(limit - 1);
  for (int c = 0; c < 100; ++c)
    CHECK(run_region() == limit);
  CHECK(pomp_close_pool() == POMP_OK);
  host_stop();
  printf("lateopen: errors %ld\n", errors);
}

// -----------------------------------------------------------------------------
// S15: workers leave and enter again as the same slots, over and over, while
// the leader's regions make them sleep in joins and barriers and be posted to.
// A slot's sleep is initialized by its first entry only: a later entry must
// find it ready, and a waker still posting to the previous occupant must not
// find it re-initialized.
// -----------------------------------------------------------------------------

static void rejoin_body(int32_t *gtid, int32_t *tid) {
  // A late member, so that others sleep with POMP_SPIN=0.
  if (*tid == omp_get_num_threads() - 1) {
    struct timespec ts = {0, 100000};
    nanosleep(&ts, NULL);
  }
  __kmpc_barrier(NULL, *gtid);
  atomic_fetch_add(&region_sum, *tid);
}

static void test_rejoin(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_pool(0) == POMP_OK);
    host_enter_all();
    wait_pool_size(limit - 1);
    for (int r = 0; r < 3; ++r) {
      atomic_store(&region_sum, 0);
      __kmpc_fork_call(NULL, 0, (void *)rejoin_body);
      CHECK(atomic_load(&region_sum) == (long)limit * (limit - 1) / 2);
    }
    // Close without waiting: the next cycle's entries race with the leaving
    // threads of this one, as the same slots.
    CHECK(pomp_close_pool() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      while (atomic_load(&host[i].done) <= c)
        sched_yield();
  }
  host_stop();
  printf("rejoin: %ld cycles, errors %ld\n", cycles, errors);
}

int main(int argc, char **argv) {
  const char *test = argc > 1 ? argv[1] : "reuse";
  long n = argc > 2 ? atol(argv[2]) : 200;
  if (!strcmp(test, "nolimit")) {
    // Run without OMP_THREAD_LIMIT; argv[3] is the limit to open with.
    test_nolimit(n, argc > 3 ? atoi(argv[3]) : 4);
    return errors ? 1 : 0;
  }
  if (!strcmp(test, "lateopen")) {
    // Run without OMP_THREAD_LIMIT; argv[2] is the limit to open with.
    test_lateopen(argc > 2 ? atoi(argv[2]) : 4);
    return errors ? 1 : 0;
  }
  if (pomp_get_thread_limit() == 0) {
    fprintf(stderr, "no thread limit: set OMP_THREAD_LIMIT\n");
    return 1;
  }
  limit = pomp_get_thread_limit();
  if (!strcmp(test, "fresh"))
    test_fresh(n);
  else if (!strcmp(test, "reuse"))
    test_reuse(n);
  else if (!strcmp(test, "server"))
    test_server(n);
  else if (!strcmp(test, "late"))
    test_late(n);
  else if (!strcmp(test, "swap"))
    test_swap(n);
  else if (!strcmp(test, "overlap"))
    test_overlap(n);
  else if (!strcmp(test, "sleepers"))
    test_sleepers(n);
  else if (!strcmp(test, "errors"))
    test_errors();
  else if (!strcmp(test, "tworoots"))
    test_tworoots(n);
  else if (!strcmp(test, "nest"))
    test_nest(n);
  else if (!strcmp(test, "sleepjoin"))
    test_sleepjoin(n);
  else if (!strcmp(test, "steal"))
    test_steal(n);
  else if (!strcmp(test, "rejoin"))
    test_rejoin(n);
  else if (!strcmp(test, "latency"))
    test_latency(n);
  else {
    fprintf(stderr, "unknown test %s\n", test);
    return 1;
  }
  return errors ? 1 : 0;
}
