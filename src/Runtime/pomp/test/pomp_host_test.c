/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ pomp_host_test.c ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Tests of the pomp host protocol: open, donate, run, release, reclaim, as a
// Java host would drive it. Each scenario of POMP.md "Host protocol" has a
// test here.
//
// Usage: pomp_host_test <test> [cycles]
//   fresh    : S1, new threads each cycle, joined after release
//   reuse    : S2, the same threads donate again each cycle
//   server   : S3, donate once, many inferences, release once
//   late     : S4, donations racing with release, and after it
//   swap     : S5, threads change slots and roles between cycles
//   overlap  : S6, next cycle opened while the last one still leaves
//   sleepers : S7, release while every worker sleeps (POMP_SPIN=0)
//   errors   : S8, misuse returns errors
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

static void wait_donated(int n) {
  while (pomp_num_donated() < n)
    sched_yield();
}

// -----------------------------------------------------------------------------
// Donor threads
// -----------------------------------------------------------------------------

// A donor that donates once, then exits: the "fresh threads" host.
static void *donor_once(void *arg) {
  int rc = pomp_donate_thread((int)(intptr_t)arg);
  CHECK(rc == POMP_OK || rc == POMP_ERR_CLOSED);
  return (void *)(intptr_t)rc;
}

// A persistent host thread: donates whenever told to, until told to stop.
// This is the "thread pool reused by the host" pattern.
typedef struct {
  pthread_t th;
  int slot;
  atomic_int go;   /* Host's command counter. */
  atomic_int done; /* Donations completed. */
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
    atomic_store(&h->last_rc, pomp_donate_thread(h->slot));
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

static void host_donate_all(void) {
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
    CHECK(pomp_open_workers() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      pthread_create(&th[i], NULL, donor_once, (void *)(intptr_t)i);
    wait_donated(limit - 1);
    for (int r = 0; r < 5; ++r)
      CHECK(run_region() == limit);
    CHECK(pomp_release_workers() == POMP_OK);
    // Joining is safe right away: release ends every donation.
    for (int i = 1; i < limit; ++i)
      pthread_join(th[i], NULL);
    CHECK(pomp_num_donated() == 0);
  }
  printf("fresh: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S2: the same host threads donate again every cycle.
// -----------------------------------------------------------------------------

static void test_reuse(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_workers() == POMP_OK);
    host_donate_all();
    wait_donated(limit - 1);
    for (int r = 0; r < 5; ++r)
      CHECK(run_region() == limit);
    CHECK(pomp_release_workers() == POMP_OK);
    // The host waits for its own threads (not the leader): reclaim.
    CHECK(pomp_reclaim_workers() == POMP_OK);
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
// S3: server mode: donate once, many inferences, release once.
// -----------------------------------------------------------------------------

static void test_server(long inferences) {
  pthread_t th[64];
  CHECK(pomp_open_workers() == POMP_OK);
  for (int i = 1; i < limit; ++i)
    pthread_create(&th[i], NULL, donor_once, (void *)(intptr_t)i);
  wait_donated(limit - 1);
  for (long k = 0; k < inferences; ++k)
    for (int r = 0; r < 3; ++r)
      CHECK(run_region() == limit);
  CHECK(pomp_release_workers() == POMP_OK);
  for (int i = 1; i < limit; ++i)
    pthread_join(th[i], NULL);
  printf("server: %ld inferences, errors %ld\n", inferences, errors);
}

// -----------------------------------------------------------------------------
// S4: donations racing with release, and arriving after it.
// -----------------------------------------------------------------------------

static void test_late(long cycles) {
  pthread_t th[64];
  long refused = 0, served = 0;
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_workers() == POMP_OK);
    // Start donors and release without waiting for them: each donation
    // either is claimed by the release or refused; none may stay.
    for (int i = 1; i < limit; ++i)
      pthread_create(&th[i], NULL, donor_once, (void *)(intptr_t)i);
    if (c & 1)
      run_region(); // Uses whoever has arrived.
    CHECK(pomp_release_workers() == POMP_OK);
    for (int i = 1; i < limit; ++i) {
      void *rc;
      pthread_join(th[i], &rc); // Must not hang.
      if ((intptr_t)rc == POMP_ERR_CLOSED)
        ++refused;
      else
        ++served;
    }
    CHECK(pomp_num_donated() == 0);
    // A donation after the release, before the next open: refused at once.
    CHECK(pomp_donate_thread(1) == POMP_ERR_CLOSED);
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
  // A thread that donated before now acts as the initial thread.
  for (int r = 0; r < 5; ++r)
    run_region();
  atomic_store(&leader_turn_done, 1);
  return NULL;
}

static void test_swap(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    // Rotate slots: host thread i donates as slot (i + c) mod (limit-1) + 1.
    for (int i = 1; i < limit; ++i)
      host[i].slot = (int)((i - 1 + c) % (limit - 1)) + 1;
    CHECK(pomp_open_workers() == POMP_OK);
    host_donate_all();
    wait_donated(limit - 1);
    CHECK(run_region() == limit);
    CHECK(pomp_release_workers() == POMP_OK);
    CHECK(pomp_reclaim_workers() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      while (atomic_load(&host[i].done) != c + 1)
        sched_yield();
  }
  host_stop();
  // A former donor as the initial thread: a fresh thread that ran
  // pomp_donate_thread before.
  {
    pthread_t t;
    pthread_create(&t, NULL, donor_once, (void *)(intptr_t)1);
    pthread_join(t, NULL); // Pool is closed: refused, returns at once.
    pthread_create(&t, NULL, leader_turn, NULL);
    pthread_join(t, NULL);
    CHECK(atomic_load(&leader_turn_done) == 1);
  }
  printf("swap: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S6: next cycle opened and donated while the last one is still leaving.
// -----------------------------------------------------------------------------

static void test_overlap(long cycles) {
  host_start();
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_workers() == POMP_OK);
    host_donate_all();
    // Do not wait for everyone: regions take whoever is idle.
    for (int r = 0; r < 3; ++r)
      run_region();
    CHECK(pomp_release_workers() == POMP_OK);
    // No reclaim: the next cycle starts while workers still leave. Each host
    // thread donates again only after its previous donation returned.
    for (int i = 1; i < limit; ++i)
      while (atomic_load(&host[i].done) != c + 1)
        sched_yield();
  }
  CHECK(pomp_reclaim_workers() == POMP_OK);
  host_stop();
  printf("overlap: %ld cycles, errors %ld\n", cycles, errors);
}

// -----------------------------------------------------------------------------
// S7: release while every worker sleeps (run with POMP_SPIN=0).
// -----------------------------------------------------------------------------

static void test_sleepers(long cycles) {
  pthread_t th[64];
  for (long c = 0; c < cycles; ++c) {
    CHECK(pomp_open_workers() == POMP_OK);
    for (int i = 1; i < limit; ++i)
      pthread_create(&th[i], NULL, donor_once, (void *)(intptr_t)i);
    wait_donated(limit - 1);
    run_region();
    // Give the workers time to fall asleep.
    struct timespec ts = {0, 2000000};
    nanosleep(&ts, NULL);
    CHECK(pomp_release_workers() == POMP_OK);
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
  if (*tid == 0)
    atomic_store(&inner_rc, pomp_release_workers());
}

static void *donate_twice(void *arg) {
  (void)arg;
  return (void *)(intptr_t)pomp_donate_thread(1);
}

static void test_errors(void) {
  pthread_t a, b;
  void *rc;
  CHECK(pomp_open_workers() == POMP_OK);
  CHECK(pomp_donate_thread(0) == POMP_ERR_SLOT);
  CHECK(pomp_donate_thread(limit) == POMP_ERR_SLOT);
  if (limit > 2) {
    pthread_create(&a, NULL, donor_once, (void *)(intptr_t)1);
    wait_donated(1);
    // Slot 1 is in use and not released.
    pthread_create(&b, NULL, donate_twice, NULL);
    pthread_join(b, &rc);
    CHECK((intptr_t)rc == POMP_ERR_SLOT);
    // Release from inside a region is refused.
    __kmpc_fork_call(NULL, 0, (void *)release_inside);
    CHECK(atomic_load(&inner_rc) == POMP_ERR_STATE);
    CHECK(pomp_release_workers() == POMP_OK);
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
    CHECK(pomp_open_workers() == POMP_OK);
    host_donate_all();
    wait_donated(limit - 1);
    run_region();
    t0 = now();
    CHECK(pomp_release_workers() == POMP_OK);
    t1 = now();
    run_region(); // Closed: leader alone, must not wait for anyone.
    t2 = now();
    t_rel += t1 - t0;
    t_next += t2 - t1;
    if (t1 - t0 > t_rel_max)
      t_rel_max = t1 - t0;
    CHECK(pomp_reclaim_workers() == POMP_OK);
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

int main(int argc, char **argv) {
  const char *test = argc > 1 ? argv[1] : "reuse";
  long n = argc > 2 ? atol(argv[2]) : 200;
  if (pomp_init() != POMP_OK) {
    fprintf(stderr, "pomp_init failed: set OMP_THREAD_LIMIT\n");
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
  else if (!strcmp(test, "latency"))
    test_latency(n);
  else {
    fprintf(stderr, "unknown test %s\n", test);
    return 1;
  }
  return errors ? 1 : 0;
}
