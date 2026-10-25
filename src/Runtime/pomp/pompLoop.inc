/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompLoop.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: static loop
// partitioning, as libomp's.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Static loops
//
// Static loop partitioning, matching libomp's (kmp_sched.cpp) results for
// kmp_sch_static and kmp_sch_static_chunked. For kmp_sch_static, libomp's
// effective flavor is "balanced": __kmp_do_serial_initialize overrides the
// "greedy" initializer of __kmp_static.
//
// T is the IV type, ST its signed type, and UT its unsigned type. All
// arithmetic is in UT, where it wraps, so the results are libomp's bit for bit
// without signed overflow. Two cases differ, both at the ends of the IV type,
// which compiled code does not reach as it normalizes its loops to
// [0, trip count - 1]:
// - A loop over all values of the type has a trip count that wraps to 0.
//   libomp then divides by zero (chunked) or splits nothing; pomp runs the
//   whole loop on tid 0.
// - The empty share of a thread is libomp's upper bound plus one step, unless
//   that wraps around the end of the type, which would make it the whole
//   range; then the upper bound moves one step back instead.
// -----------------------------------------------------------------------------

// Give the calling thread an empty share of the loop that ends at `hi`.
template <class T, class ST, class UT>
void emptyShare(T *plower, T *pupper, UT hi, ST incr) {
  T past = T(incr > 0 ? hi + 1 : hi - 1);
  if (incr > 0 ? past > *pupper : past < *pupper) {
    *plower = past;
  } else {
    *plower = *pupper;
    *pupper = T(incr > 0 ? hi - 1 : hi + 1);
  }
}

template <class T, class ST, class UT>
void staticInit(int32_t sched, int32_t *plast, T *plower, T *pupper,
    ST *pstride, ST incr, ST chunk) {
  RegionStack *regions = Thread::currentRegions();
  UT lo = UT(*plower), hi = UT(*pupper), inc = UT(incr);
  if (incr == 0)
    fatal("loop increment of 0");
  sched &= ~kSchModifierMask;
  if (incr > 0 ? (*pupper < *plower) : (*plower < *pupper)) {
    if (plast)
      *plast = 0;
    *pstride = incr;
    return;
  }
  UT nth = regions ? UT(regions->topNth()) : 1;
  UT tid = regions ? UT(regions->topTid()) : 0;
  if (nth == 1) {
    if (plast)
      *plast = 1;
    *pstride = ST(incr > 0 ? hi - lo + 1 : hi - lo - 1);
    return;
  }
  UT trip;
  if (incr == 1)
    trip = hi - lo + 1;
  else if (incr == -1)
    trip = lo - hi + 1;
  else if (incr > 0)
    trip = (hi - lo) / inc + 1;
  else
    trip = (lo - hi) / (UT(0) - inc) + 1;
  if (sched != kSchStatic && sched != kSchStaticChunked)
    fatal("unsupported loop schedule (only static is implemented)");
  if (trip == 0) { // All values of the type: the whole loop on tid 0.
    if (tid != 0)
      emptyShare(plower, pupper, hi, incr);
    if (plast)
      *plast = tid == 0;
    *pstride = ST(hi - lo);
    return;
  }
  if (sched == kSchStatic) {
    if (trip < nth) {
      if (tid < trip)
        *pupper = *plower = T(lo + tid * inc);
      else
        emptyShare(plower, pupper, hi, incr);
      if (plast)
        *plast = (tid == trip - 1);
    } else {
      UT small = trip / nth, extras = trip % nth;
      lo += inc * (tid * small + (tid < extras ? tid : extras));
      *plower = T(lo);
      *pupper = T(lo + small * inc - (tid < extras ? 0 : inc));
      if (plast)
        *plast = (tid == nth - 1);
    }
    *pstride = ST(trip);
  } else {
    UT c = chunk < 1 ? 1 : UT(chunk);
    if (c > trip)
      c = trip;
    UT nchunks = trip / c + (trip % c ? 1 : 0);
    UT span = c * inc;
    if (nchunks < nth) {
      *pstride = ST(span * nchunks);
      if (tid < nchunks) {
        lo += span * tid;
        *plower = T(lo);
        *pupper = T(lo + span - inc);
      } else {
        emptyShare(plower, pupper, hi, incr);
      }
    } else {
      *pstride = ST(span * nth);
      lo += span * tid;
      *plower = T(lo);
      *pupper = T(lo + span - inc);
    }
    if (plast)
      *plast = (tid == (nchunks - 1) % nth);
  }
}
