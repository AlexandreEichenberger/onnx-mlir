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
#define POMP_ERR_LIMIT 1  /* Thread limit missing, out of range, or changed. */
#define POMP_ERR_SLOT 2   /* Slot number out of range or already in use. */
#define POMP_ERR_STATE 3  /* Call not valid in the current state. */
#define POMP_ERR_CLOSED 4 /* Entry refused: the pool is closed. */
#define POMP_ERR_NOMEM 5  /* The thread limit's descriptors cannot be allocated. */

//===----------------------------------------------------------------------===//
// General interface description

// The pool is the set of host threads lent to pomp, the pool threads. A host
// (e.g. Java) uses it in cycles, from any number of host threads:
//
//   pomp_set_thread_limit(n)    once per process, before the first open
//   pomp_open_pool()            leader: open (once per cycle; see below)
//   pomp_enter_pool(slot)       each pool thread: work until the pool closes
//   ... run the model ...       leader: parallel regions use the pool
//   pomp_close_pool()           leader: close; pool threads return
//   pomp_wait_pool_empty()      optional, any host thread: wait for them
//
// There is no init call. Every entry point, including pomp_enter_pool and the
// __kmpc_* ones, initializes pomp on first use: it reads the environment, and
// the thread limit from OMP_THREAD_LIMIT if it is set. Concurrent first calls
// are safe. The pool starts closed: pomp_open_pool opens it, also when
// OMP_THREAD_LIMIT gave the limit. An OMP_THREAD_LIMIT that is set but invalid
// is ignored with a warning on stderr; a host can detect it as
// pomp_get_thread_limit() == 0.
//
// Any number of host threads that are not in the pool may run parallel
// regions at once, each as the initial thread of its own regions: each
// region gets the idle pool threads it can, up to the thread limit, so that
// together they may run more threads than the limit. Each such thread has a
// small descriptor, reused by another thread once it ends.

//===----------------------------------------------------------------------===//
// Interface calls

// Set the thread limit (initial thread included), once per process; like
// OpenMP's thread-limit-var, it never changes afterwards. A valid
// OMP_THREAD_LIMIT sets it first and takes precedence: the call then returns
// POMP_OK whatever `thread_limit` is, warning once on stderr if it differs.
// Otherwise the first call with `thread_limit` in [1, POMP_MAX_THREADS] sets
// it; later calls return POMP_OK with the same value and POMP_ERR_LIMIT with
// any other. Setting the limit does not open the pool. It allocates pomp's
// per-thread descriptors, once; if that fails, the call returns
// POMP_ERR_NOMEM, no limit is set, and a later call may try again.
int pomp_set_thread_limit(int thread_limit);

// The thread limit, or 0 while none is set (OMP_THREAD_LIMIT unset or
// invalid, and no successful pomp_set_thread_limit yet).
//
// omp_get_thread_limit returns this limit, and omp_get_max_threads returns at
// most this limit. While no limit is set, both return 1, and the first such
// call prints a warning on stderr: until a limit is set, the pool is closed and
// every parallel region runs on one thread. A host should therefore call
// pomp_set_thread_limit before any code calls these two to size its work.
int pomp_get_thread_limit(void);

// Open the pool: threads may enter it, and parallel regions use the threads
// that have entered. Once per cycle, the first one included. Until the first
// open, parallel regions run on one thread, with one warning on stderr.
// Returns POMP_ERR_LIMIT, with the pool left closed, while no thread limit is
// set. Must be called by a thread that is not in the pool, outside of any
// parallel region (POMP_ERR_STATE). Does not wait for anything: threads of the
// previous cycle still leaving are never used.
int pomp_open_pool(void);

// The calling thread enters the pool as pool thread `slot`, in
// [1, thread_limit - 1]; slot 0 is any thread that is not in the pool. While
// the pool is open,
// the call returns only once pomp_close_pool releases the thread, with
// POMP_OK. It returns at once with POMP_ERR_CLOSED if the pool is closed
// (including while no thread limit is set), with POMP_ERR_SLOT if another
// thread is in the pool as `slot`, and with POMP_ERR_STATE from inside a
// parallel region. If the previous thread of `slot` was released but has not
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
//   OMP_THREAD_LIMIT      as pomp_set_thread_limit
//   OMP_NUM_THREADS       "n" or "n1,n2"
//   OMP_MAX_ACTIVE_LEVELS 1 or 2
//   POMP_TREE_BARRIER     the smallest team that uses the tree barrier
//   POMP_SPIN             "auto", "auto:MAX" or a number of microseconds
// POMP_SPIN may be set at any time and applies to the next waits; the others
// only before the first pomp_open_pool (POMP_ERR_STATE after). If the
// environment variable is set, it takes precedence: the call returns POMP_OK
// (or POMP_ERR_LIMIT for an invalid value, as always) and changes nothing, unless POMP_IGNORE_ENV=1 is set, which makes calls take
// precedence over the environment (OMP_THREAD_LIMIT is then not read at all:
// the host must set the limit). POMP_ERR_LIMIT: unknown name, or invalid
// value (the setting is unchanged).
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
