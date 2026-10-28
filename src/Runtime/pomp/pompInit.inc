/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompInit.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the settings, read from the environment once, and the
// thread limit with the allocation of the descriptors.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

// `n` zeroed objects of `size` bytes, line-aligned, or nullptr.
void *allocLines(size_t n, size_t size) {
  char *p = static_cast<char *>(calloc(1, n * size + kCacheLine));
  if (!p)
    return nullptr;
  uintptr_t a = (uintptr_t(p) + kCacheLine - 1) & ~uintptr_t(kCacheLine - 1);
  return reinterpret_cast<void *>(a);
}

// Called on each entry as this slot, before it is recruitable, and at
// allocation for slot 0. Entries as a slot are serialized by `present_`.
inline void Thread::init() {
  if (!initialized_) {
    post_.init();
    initialized_ = true;
  }
}

// Concurrent first calls wait for the one that does it.
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

  // Max active levels: as libomp, 1 unless asked for or OMP_NUM_THREADS lists
  // more than one level.
  maxActive_ = n > 1 ? kMaxActive : 1;
  s = getenv("OMP_MAX_ACTIVE_LEVELS");
  if (s && (v = parseInt(s, nullptr)) >= 0)
    maxActive_ = v < 1 ? 1 : (v > kMaxActive ? kMaxActive : int(v));

  treeMin_ = kDefaultTreeBarrier;
  s = getenv("POMP_TREE_BARRIER");
  if (s && (v = parseInt(s, nullptr)) >= 0)
    treeMin_ = int(v);

  // POMP_SPIN: auto (default), auto:MAX, or a fixed number of us.
  spinUs_ = -1;
  spinMaxUs_ = kDefaultSpinMaxUs;
  s = getenv("POMP_SPIN");
  if (s) {
    while (*s == ' ' || *s == '\t')
      ++s;
    if (!strncmp(s, "auto", 4)) {
      s += 4;
      if (*s == ':') {
        v = parseInt(s + 1, &end);
        if (v >= 0 && !*end)
          spinMaxUs_ = v;
        else
          fprintf(stderr, "pomp: warning: POMP_SPIN=auto:%s is invalid; "
                          "using auto:%ld\n",
              s + 1, kDefaultSpinMaxUs);
      } else if (*s) {
        fprintf(stderr, "pomp: warning: POMP_SPIN=auto%s is invalid; using "
                        "auto\n", s);
      }
    } else if ((v = parseInt(s, &end)) >= 0 && !*end) {
      spinUs_ = v;
    } else {
      fprintf(stderr,
          "pomp: warning: POMP_SPIN=\"%s\" is invalid (auto, auto:MAX or a "
          "number of us); using auto\n",
          s);
    }
  }

  s = getenv("OMP_THREAD_LIMIT");
  if (s) {
    v = parseInt(s, &end);
    int rc = *end ? POMP_ERR_LIMIT : setLimit(v, true);
    if (rc != POMP_OK)
      fprintf(stderr,
          "pomp: warning: OMP_THREAD_LIMIT=\"%s\" is not in [1, %d] or "
          "cannot be allocated, and is ignored; the pool stays closed "
          "until pomp_set_thread_limit gives a limit\n",
          s, kMaxThreads);
  }
}

// Set the thread limit, once per process (see pomp_set_thread_limit). Once it
// is set, the same value succeeds; another one is refused if a call set the
// limit, and ignored with one warning if OMP_THREAD_LIMIT did.
int Settings::setLimit(long limit, bool fromEnv) {
  ScopedLock<Mutex> lock(limitMutex_);
  int cur = this->limit();
  if (cur) {
    if (limit == cur)
      return POMP_OK;
    if (!limitFromEnv_)
      return POMP_ERR_LIMIT;
    if (!limitMismatchSeen_.exchange(1))
      fprintf(stderr,
          "pomp: warning: thread limit %ld requested, but OMP_THREAD_LIMIT "
          "already set it to %d for this process; using %d\n",
          limit, cur, cur);
    return POMP_OK;
  }
  if (limit < 1 || limit > kMaxThreads)
    return POMP_ERR_LIMIT;
  // One block, never freed: `limit` threads, then `limit` + 1 teams from the
  // next line. On failure no limit is set, so a later call may retry.
  size_t tBytes = (size_t(limit) * sizeof(Thread) + kCacheLine - 1) &
                  ~size_t(kCacheLine - 1);
  char *p = static_cast<char *>(
      allocLines(1, tBytes + (size_t(limit) + 1) * sizeof(Team)));
  if (!p)
    return POMP_ERR_NOMEM;
  Thread *t = reinterpret_cast<Thread *>(p);
  Team *m = reinterpret_cast<Team *>(p + tBytes);
  // Slot 0 may sleep before any pool thread enters; those init on entry.
  t[0].init();
  // Set before Team::init, which reads index(). Not yet visible to other
  // threads: the limit_ store below publishes the descriptors.
  threads = t;
  teams = m;
  for (int32_t i = 0; i <= limit; ++i)
    m[i].init();
  for (int i = 0; i < kMaxActive; ++i) {
    int n = envNthreads_[i];
    nthreads_[i].store(n && n < limit ? n : int(limit));
  }
  limitSlots_.store(SlotMask::below(int(limit)));
  limitFromEnv_ = fromEnv;
  // Publishes the descriptors (see limitAcquire).
  limit_.store(int(limit), Order::Release);
  return POMP_OK;
}

void Settings::warnNoLimit() {
  if (!noLimitSeen_.exchange(1))
    fprintf(stderr,
        "pomp: warning: omp_get_max_threads/omp_get_thread_limit return 1: "
        "no thread limit is set yet (OMP_THREAD_LIMIT is unset or invalid, "
        "and no pomp_set_thread_limit(limit) with limit in [1, %d] has been "
        "called); "
        "until one is, parallel regions run on one thread\n",
        kMaxThreads);
}
