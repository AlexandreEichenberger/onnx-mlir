/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pomp.h -----------------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// pomp: a minimal pool-based OpenMP runtime that implements the subset of the
// libomp (__kmpc_*) ABI emitted for onnx-mlir models and zDNNx, with threads
// lent by the host (e.g. Java) through a pool, and returned after use.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_POMP_H
#define ONNX_MLIR_POMP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum value of OMP_THREAD_LIMIT, including the initial thread. The pool is
// described by one 64-bit word, one bit per thread slot: slot 0 is the
// initial thread of a team, which any host thread outside the pool may be;
// slots 1..limit-1 are the pool threads.
#define POMP_MAX_THREADS 64

// Return codes of the host interface.
#define POMP_OK 0
#define POMP_ERR_LIMIT 1  /* Thread limit missing or out of range. */
#define POMP_ERR_SLOT 2   /* Slot number out of range or already in use. */
#define POMP_ERR_STATE 3  /* Call not valid in the current state. */
#define POMP_ERR_CLOSED 4 /* Entry refused: the pool is closed. */
#define POMP_ERR_NOMEM 5  /* The thread limit's descr. cannot be allocated. */
#define POMP_ERR_OPEN 6   /* Open refused: the pool is already open. */

//===----------------------------------------------------------------------===//
// General interface description

// The pool is the set of host threads lent to pomp, the pool threads. A host
// (e.g. Java) uses it in cycles. One host thread, the cycle's owner, opens
// and closes each cycle; any number of host threads run models in it:
//
//   pomp_open_pool(16, &n)       owner: open; n is the thread limit
//   pomp_enter_pool(slot)        n-1 pool threads, slots 1..n-1: work until
//                                the pool closes
//   ... run models ...           any host threads: regions use the pool
//   pomp_close_pool()            owner: close; pool threads return
//   pomp_wait_pool_empty()       optional: wait until they have returned
//
// A typical host, for one cycle (error handling shortened):
//
//   static void *pool_thread(void *slot) {      // a pool thread
//     pomp_enter_pool((int)(intptr_t)slot);     // returns at the close
//     return NULL;
//   }
//
//   int n;                                      // the limit it gets:
//   int rc = pomp_open_pool(16, &n);            // OMP_THREAD_LIMIT, else 16
//   if (rc == POMP_OK) {
//     // This thread owns the cycle: it starts the pool threads.
//     for (int slot = 1; slot < n; ++slot)
//       pthread_create(&th[slot], NULL, pool_thread, (void *)(intptr_t)slot);
//     while (pomp_pool_size() < n - 1)          // optional: all of them
//       sched_yield();                          // for the first region
//   } else if (rc != POMP_ERR_OPEN) {
//     // POMP_ERR_LIMIT or POMP_ERR_NOMEM: the pool stays closed, and
//     // models run on one thread.
//   }
//   // POMP_ERR_OPEN: another thread owns the cycle; use its pool as is.
//
//   run_model();                                // any number of times
//
//   if (rc == POMP_OK) {
//     // Only the owner closes, once every thread's models are done.
//     pomp_close_pool();
//     for (int slot = 1; slot < n; ++slot)
//       pthread_join(th[slot], NULL);           // they return at once
//   }
//
// There is no init call. Every entry point, including the __kmpc_* ones,
// initializes pomp on first use: it reads the environment. Concurrent first
// calls are safe. The pool starts closed. Until it is opened, parallel regions
// run on one thread.
//
// The thread limit is OMP_THREAD_LIMIT if set, else the fallback the host
// gives its first open; it never changes then. A host should open the pool
// before model code runs, so that code sizing its work by omp_get_max_threads
// sees the right value. To lend fewer threads than the limit, a host sets
// OMP_NUM_THREADS (pomp_set_option) to the team size it will have.
//
// Any number of host threads that are not in the pool may run parallel
// regions at once, each as the initial thread of its own regions: each
// region gets the idle pool threads it can, up to the thread limit, so that
// together they may run more threads than the limit. Each such thread has a
// small descriptor (about 3 KB on s390x), allocated at its first parallel
// region and reused by another thread once it ends. If it cannot be
// allocated, pomp prints an error on stderr and aborts the process: the
// compiled code that starts a region has no way to report an error.

//===----------------------------------------------------------------------===//
// Interface calls

// Open the pool for a cycle: threads may enter it, and parallel regions use
// the threads that have entered. `*limit` returns the thread limit, initial
// thread included: OMP_THREAD_LIMIT if set; else `fallback_limit`, which the
// first successful open fixes for the process (later fallbacks are ignored).
// `fallback_limit` is in [1, POMP_MAX_THREADS], or 0 for none: the open then
// needs OMP_THREAD_LIMIT, or an earlier open.
//
// POMP_OK: this call opened the pool. Its caller owns the cycle: it starts
// `*limit` - 1 pool threads, on slots 1..*limit-1, and closes the cycle.
// POMP_ERR_OPEN: the pool is already open; nothing changes, and `*limit`
// returns the limit. Such a caller should not start pool threads nor close:
// the cycle is another caller's.
// POMP_ERR_LIMIT: `limit` is null, `fallback_limit` is out of range, or no
// limit is known; the pool stays closed.
// POMP_ERR_NOMEM: the first open cannot allocate pomp's descriptors; the pool
// stays closed, its fallback is not kept, and a later open may try again.
// POMP_ERR_STATE: called from a pool thread or inside a parallel region.
// Does not wait for anything: threads of the previous cycle still leaving are
// never used.
int pomp_open_pool(int fallback_limit, int *limit);

// The thread limit: OMP_THREAD_LIMIT if set, else the fallback of the first
// successful open; 0 while neither is known. It is what omp_get_thread_limit
// returns; until it is known, omp_get_thread_limit and omp_get_max_threads
// return 1, and the first such call prints a warning on stderr.
int pomp_get_thread_limit(void);

// The calling thread enters the pool as pool thread `slot`, in
// [1, limit - 1]; slot 0 is any thread that is not in the pool. While
// the pool is open, the call returns only once pomp_close_pool releases the
// thread, with POMP_OK. It returns at once with POMP_ERR_CLOSED if the pool is
// closed (including while no thread limit is set), with POMP_ERR_SLOT if
// another thread is in the pool as `slot`, and with POMP_ERR_STATE from inside
// a parallel region. If the previous thread of `slot` was released but has not
// left yet, the call waits for it to leave. A thread may enter again after it
// returns, as the same slot or another one, and may also run parallel regions
// itself, before and after.
int pomp_enter_pool(int slot);

// Close the pool and release every thread in it; return at once, without
// waiting for them (see pomp_wait_pool_empty). Must be called by a thread
// that is not in the pool, outside of any parallel region (POMP_ERR_STATE).
// Threads in a team when it is called are released when that team ends.
// Until the next pomp_open_pool, pomp_enter_pool returns POMP_ERR_CLOSED and
// parallel regions run on their initial thread alone. Each released thread
// returns from pomp_enter_pool as soon as it notices; sleeping ones are woken
// by the threads that leave first.
int pomp_close_pool(void);

// Wait until every released thread has returned from pomp_enter_pool. Never
// needed for correctness; for a host thread that wants to join or reuse them.
// May be called by any thread that is not in the pool.
int pomp_wait_pool_empty(void);

// Nonzero while the pool is open.
int pomp_is_pool_open(void);

// Number of threads in the pool: counted once accepted, uncounted once
// returning to the host.
int pomp_pool_size(void);

// Set one of pomp's settings from code, for hosts that cannot set
// environment variables (a JVM, for one). `name` is the setting's environment
// variable and `value` takes the same strings:
//   OMP_NUM_THREADS       "n" or "n1,n2"
//   OMP_MAX_ACTIVE_LEVELS 1 or 2
//   POMP_TREE_BARRIER     the smallest team that uses the tree barrier
//   POMP_SPIN             "auto", "auto:MAX" or a number of microseconds
// POMP_SPIN may be set at any time and applies to the next waits; the others
// only before the first pomp_open_pool (POMP_ERR_STATE after). As in OpenMP,
// the environment variable, if set, only gives the initial value, and a call
// overrides it. The thread limit is not one of them: pomp_open_pool sets it.
// POMP_ERR_LIMIT: unknown name, or invalid value (the setting is unchanged).
int pomp_set_option(const char *name, const char *value);

// Diagnosis: copy up to `n` of pomp's sleep and wake-up counters into
// `counts` and return how many; 0 unless pomp is built with -DPOMP_STATS. In
// order: regions forked with workers, idle sleeps, team-wait sleeps, posts,
// posts at forks, ns spent posting, ns spent asleep, root blocks allocated
// (one per host thread running regions at once; ended ones are reused).
int pomp_get_stats(unsigned long long *counts, int n);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_POMP_H
