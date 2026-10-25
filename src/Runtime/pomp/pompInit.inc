/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompInit.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: the settings, read
// from the environment once, and the thread limit with the allocation of the
// descriptors.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

// `n` zeroed objects of `size` bytes, starting on a cache line, or nullptr.
// calloc's alignment may be less than a line: over-allocate and round up.
void *allocLines(size_t n, size_t size) {
  char *p = static_cast<char *>(calloc(1, n * size + kCacheLine));
  if (!p)
    return nullptr;
  uintptr_t a = (uintptr_t(p) + kCacheLine - 1) & ~uintptr_t(kCacheLine - 1);
  return reinterpret_cast<void *>(a);
}

// Set the thread limit, once per process, as OpenMP's thread-limit-var: from
// OMP_THREAD_LIMIT at initialization, or else from the first open that gives
// one. Once set it never changes: a later request for another value (0 means
// none) is clamped to it, with one warning per process, and succeeds. Fails
// only while no limit is set and `limit` is not a valid one.
int Settings::setLimit(long limit) {
  ScopedLock<Mutex> lock(limitMutex_);
  int cur = this->limit();
  if (cur) {
    if (limit != 0 && limit != cur && !limitMismatchSeen_.exchange(1))
      fprintf(stderr,
          "pomp: warning: thread limit %ld requested, but the limit is "
          "already %d for this process; using %d\n",
          limit, cur, cur);
    return POMP_OK;
  }
  if (limit < 1 || limit > kMaxThreads)
    return POMP_ERR_LIMIT;
  Thread *t = static_cast<Thread *>(allocLines(size_t(limit), sizeof(Thread)));
  Team *m = static_cast<Team *>(allocLines(size_t(limit) + 1, sizeof(Team)));
  if (!t || !m) {
    // Not freed: aligned blocks keep no base pointer, and this happens once.
    return POMP_ERR_NOMEM;
  }
  // The root sleeps in joins and barriers before any worker enters: its
  // sleep is ready now. Workers ready theirs on entry.
  t[0].initPost();
  threads = t;
  teams = m;
  for (int i = 0; i < kMaxActive; ++i) {
    int n = envNthreads_[i];
    nthreads_[i].store(n && n < limit ? n : int(limit));
  }
  limitSlots_.store(SlotMask::below(int(limit)));
  // Publishes the descriptors: read them only after an acquire of the limit
  // (limitAcquire), or after a step that follows one.
  limit_.store(int(limit), Order::Release);
  return POMP_OK;
}

void Settings::init() {
  const char *s, *end;
  long v;

  limitMutex_.init();
  SelfWord::createStorage();

  // OMP_NUM_THREADS: "n" or "n1,n2"; missing entries repeat the last one.
  // Clamped to the limit once it is known.
  int n = 0;
  s = getenv("OMP_NUM_THREADS");
  while (s && n < kMaxActive) {
    v = parseInt(s, &end);
    if (v < 1)
      break;
    envNthreads_[n++] = v > kMaxThreads ? kMaxThreads : int(v);
    if (*end != ',')
      break;
    s = end + 1;
  }
  for (int i = n > 0 ? n : kMaxActive; i < kMaxActive; ++i)
    envNthreads_[i] = envNthreads_[i - 1];

  // Max active levels: as in libomp, 1 unless asked for, or unless
  // OMP_NUM_THREADS lists more than one level.
  maxActive_ = n > 1 ? kMaxActive : 1;
  s = getenv("OMP_MAX_ACTIVE_LEVELS");
  if (s && (v = parseInt(s, nullptr)) >= 0)
    maxActive_ = v < 1 ? 1 : (v > kMaxActive ? kMaxActive : int(v));

  treeMin_ = kDefaultTreeBarrier;
  s = getenv("POMP_TREE_BARRIER");
  if (s && (v = parseInt(s, nullptr)) >= 0)
    treeMin_ = int(v);

  spin_ = kDefaultSpin;
  s = getenv("POMP_SPIN");
  if (s && (v = parseInt(s, nullptr)) >= 0)
    spin_ = v;

  // OMP_THREAD_LIMIT, if set, must be valid. If it is not set, or not valid,
  // the pool starts closed, and the first open that gives a limit opens it.
  s = getenv("OMP_THREAD_LIMIT");
  if (s) {
    v = parseInt(s, &end);
    int rc = v < 1 || *end ? POMP_ERR_LIMIT : setLimit(v);
    if (rc != POMP_OK)
      fprintf(stderr,
          "pomp: warning: OMP_THREAD_LIMIT=\"%s\" is not in [1, %d] or "
          "cannot be allocated, and is ignored; the pool stays closed "
          "until pomp_open_pool gives a limit\n",
          s, kMaxThreads);
  }
  if (!limit())
    pool.closeGate();
}

// Concurrent first calls wait for the one that does it. Without a valid
// thread limit, the pool stays closed and regions run on one thread: nothing
// here is fatal.
void Settings::initOnce() {
  int expected = 0;
  if (initialized_.compareExchange(expected, 1)) {
    init();
    initialized_.store(2, Order::Release);
  } else {
    while (initialized_.load(Order::Acquire) != 2)
      sched_yield();
  }
}
