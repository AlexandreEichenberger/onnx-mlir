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
// threads donated by the host (e.g. Java) and released after use.
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
#define POMP_ERR_LIMIT 1  /* OMP_THREAD_LIMIT missing or out of range. */
#define POMP_ERR_SLOT 2   /* Slot number out of range or already in use. */
#define POMP_ERR_STATE 3  /* Call not valid in the current state. */
#define POMP_ERR_CLOSED 4 /* Donation refused: pool closed by a release. */

// Initialize the runtime. Reads OMP_THREAD_LIMIT, which must be set to a value
// in [1, POMP_MAX_THREADS]. Called implicitly on the first fork if not called
// explicitly; an explicit call lets the host report the error (e.g. throw).
// The pool starts open.
int pomp_init(void);

// Return the thread limit (initial thread included), or 0 before init.
int pomp_get_thread_limit(void);

// Donate the calling thread to the pool as worker `slot`, in
// [1, thread_limit - 1]; slot 0 is the initial thread. While the pool is open,
// the call returns only once a release claims the thread, with POMP_OK. It
// returns at once with POMP_ERR_CLOSED if the pool is closed, and with
// POMP_ERR_SLOT if another thread is donated as `slot` and not released. If
// the previous donor of `slot` was released but has not left yet, the call
// waits for it to leave. A thread may donate again after it returns, with the
// same slot or another one, and may also act as the initial thread.
int pomp_donate_thread(int slot);

// Close the pool and release every donated worker; return at once, without
// waiting for the workers (see pomp_reclaim_workers). Must be called by the
// initial thread outside of any parallel region. Until pomp_open_workers,
// donations return POMP_ERR_CLOSED and parallel regions run on the initial
// thread alone. Each released worker returns from pomp_donate_thread as soon
// as it notices; sleeping ones are woken by the workers that leave first.
int pomp_release_workers(void);

// Wait until every released worker has returned from pomp_donate_thread.
// Never needed for correctness; for a host thread that wants to join or reuse
// the donors. May be called by any thread that is not a donated worker.
int pomp_reclaim_workers(void);

// Reopen the pool after a release: donations are accepted again and parallel
// regions use the workers donated from now on. Workers of the previous cycle
// still leaving are never recruited. Must be called by the initial thread
// outside of any parallel region. Does not wait for anything.
int pomp_open_workers(void);

// Nonzero while the pool is open.
int pomp_is_open(void);

// Number of workers currently donated: counted once accepted, uncounted once
// returning to the host.
int pomp_num_donated(void);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_POMP_H
