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
// libomp (__kmpc_*) ABI emitted for onnx-mlir models and zDNNx, with worker
// threads lent by the host (e.g. Java) through a pool, and returned after use.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_POMP_H
#define ONNX_MLIR_POMP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum value of OMP_THREAD_LIMIT, including the initial thread. The pool is
// described by one 64-bit word, one bit per thread slot.
#define POMP_MAX_THREADS 64

// Return codes of the host interface.
#define POMP_OK 0
#define POMP_ERR_LIMIT 1  /* Thread limit missing, out of range, or changed. */
#define POMP_ERR_SLOT 2   /* Slot number out of range or already in use. */
#define POMP_ERR_STATE 3  /* Call not valid in the current state. */
#define POMP_ERR_CLOSED 4 /* Entry refused: the pool is closed. */

// The pool is the set of host threads lent to pomp as workers. A host
// (e.g. Java) uses it in cycles, from any number of host threads:
//
//   pomp_open_pool(limit)       leader: open (once per cycle; see below)
//   pomp_enter_pool(slot)       each lent thread: work until the pool closes
//   ... run the model ...       leader: parallel regions use the pool
//   pomp_close_pool()           leader: close; lent threads return
//   pomp_wait_pool_empty()      optional, any host thread: wait for them
//
// There is no init call. Every entry point, including pomp_enter_pool and the
// __kmpc_* ones, initializes pomp on first use: it reads the environment, and
// the thread limit from OMP_THREAD_LIMIT if it is set. Concurrent first calls
// are safe. The pool starts open if OMP_THREAD_LIMIT gave the limit, and closed
// otherwise. An OMP_THREAD_LIMIT that is set but invalid is ignored with a
// warning on stderr; a host can detect it as pomp_get_thread_limit() == 0.

// The thread limit (initial thread included, in [1, POMP_MAX_THREADS]), or 0
// while no limit is set (OMP_THREAD_LIMIT unset or invalid, and no
// pomp_open_pool with a limit yet).
int pomp_get_thread_limit(void);

// Open the pool: threads may enter it, and parallel regions use the threads
// that have entered. Once per cycle, after pomp_close_pool; not needed for the
// first cycle if OMP_THREAD_LIMIT is set, since the pool then starts open.
// Must be called by a thread that is not in the pool, outside of any parallel
// region (POMP_ERR_STATE). Does not wait for anything: threads of the previous
// cycle still leaving are never used.
//
// The thread limit is set once per process and never changes, like OpenMP's
// thread-limit-var: by OMP_THREAD_LIMIT if it is set and valid, or else by
// the first pomp_open_pool with a `thread_limit` in [1, POMP_MAX_THREADS].
// Once it is set, `thread_limit` may be 0 (use it); any other value is
// clamped to it, with one warning on stderr per process, and the call still
// opens the pool. Returns POMP_ERR_LIMIT, with the pool left as it was, only
// while no limit is set and `thread_limit` is not a valid one.
int pomp_open_pool(int thread_limit);

// The calling thread enters the pool as worker `slot`, in
// [1, thread_limit - 1]; slot 0 is the initial thread. While the pool is open,
// the call returns only once pomp_close_pool releases the thread, with
// POMP_OK. It returns at once with POMP_ERR_CLOSED if the pool is closed
// (including while no thread limit is set), with POMP_ERR_SLOT if another
// thread is in the pool as `slot`, and with POMP_ERR_STATE from inside a
// parallel region. If the previous thread of `slot` was released but has not
// left yet, the call waits for it to leave. A thread may enter again after it
// returns, as the same slot or another one, and may also act as the initial
// thread.
int pomp_enter_pool(int slot);

// Close the pool and release every thread in it; return at once, without
// waiting for them (see pomp_wait_pool_empty). Must be called by a thread
// that is not in the pool, outside of any parallel region (POMP_ERR_STATE).
// Threads in a team when it is called are released when that team ends.
// Until the next pomp_open_pool, pomp_enter_pool returns POMP_ERR_CLOSED and
// parallel regions run on the initial thread alone. Each released thread
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

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_POMP_H
