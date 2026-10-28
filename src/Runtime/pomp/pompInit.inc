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
  limitMutex_.init();
  optionMutex_.init();
  SelfWord::createStorage();

  // Defaults; then each setting the environment sets.
  maxActive_.store(1, Order::Relaxed);
  treeMin_.store(kDefaultTreeBarrier, Order::Relaxed);
  spinUs_.store(-1, Order::Relaxed);
  spinMaxUs_.store(kDefaultSpinMaxUs, Order::Relaxed);
  const char *s = getenv("POMP_IGNORE_ENV");
  ignoreEnv_ = s && *s && strcmp(s, "0") != 0;
  // OMP_DISPLAY_ENV: TRUE or VERBOSE (any case), as in OpenMP.
  s = getenv("OMP_DISPLAY_ENV");
  if (s && (!strcasecmp(s, "true") || !strcmp(s, "1")))
    displayEnv_ = 1;
  else if (s && !strcasecmp(s, "verbose"))
    displayEnv_ = 2;
  // OMP_NUM_THREADS first: it gives OMP_MAX_ACTIVE_LEVELS its default.
  static const char *const names[] = {"OMP_NUM_THREADS",
      "OMP_MAX_ACTIVE_LEVELS", "POMP_TREE_BARRIER", "POMP_SPIN"};
  for (const char *name : names)
    if ((s = getenv(name)) && setOption(name, s, kEnv) != POMP_OK)
      fprintf(stderr, "pomp: warning: %s=\"%s\" is invalid and ignored\n",
          name, s);

  // Under POMP_IGNORE_ENV the host sets the limit itself: the limit cannot
  // change once set, so OMP_THREAD_LIMIT is not used at all.
  const char *end;
  long v;
  s = ignoreEnv_ ? nullptr : getenv("OMP_THREAD_LIMIT");
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

// The settings pomp_set_option knows, in the order of sources_. Whether each
// may change once the pool has opened: team sizes and the barrier kind are
// read at forks, so only before; POMP_SPIN at every wait, so at any time.
namespace {
struct OptionDesc {
  const char *name;
  bool anytime;
  bool (Settings::*set)(const char *, bool);
};
} // namespace

int Settings::setOption(const char *name, const char *value, Source from) {
  static const OptionDesc kOptions[kNumOptions] = {
      {"OMP_NUM_THREADS", false, &Settings::setNumThreads},
      {"OMP_MAX_ACTIVE_LEVELS", false, &Settings::setMaxActive},
      {"POMP_TREE_BARRIER", false, &Settings::setTreeMin},
      {"POMP_SPIN", true, &Settings::setSpin},
  };
  if (!name || !value)
    return POMP_ERR_LIMIT;
  if (!strcmp(name, "OMP_THREAD_LIMIT")) {
    const char *end;
    long v = parseInt(value, &end);
    return *end ? POMP_ERR_LIMIT : setLimit(v, false);
  }
  int k = 0;
  while (k < kNumOptions && strcmp(name, kOptions[k].name))
    ++k;
  if (k == kNumOptions)
    return POMP_ERR_LIMIT;
  if (from == kCall && !kOptions[k].anytime && pool.wasOpened())
    return POMP_ERR_STATE;
  ScopedLock<Mutex> lock(optionMutex_);
  // A set environment variable wins over calls, unless POMP_IGNORE_ENV. An
  // overridden call still validates its value, and succeeds if it is valid:
  // it did not fail, it was overridden.
  bool overridden = (from == kCall && sources_[k] == kEnv && !ignoreEnv_) ||
                    (from == kEnv && ignoreEnv_ && sources_[k] == kCall);
  if (!(this->*kOptions[k].set)(value, !overridden))
    return POMP_ERR_LIMIT;
  if (overridden)
    return POMP_OK;
  sources_[k] = from;
  return POMP_OK;
}

namespace {
const char *sourceName(Settings::Source s) {
  return s == Settings::kEnv ? "environment"
         : s == Settings::kCall ? "pomp_set_option"
                                : "default";
}
} // namespace

// In OpenMP's OMP_DISPLAY_ENV format, each line followed by where the value
// comes from.
void Settings::displayOnce() {
  if (!displayEnv_ || displayed_.exchange(1))
    return;
  ScopedLock<Mutex> lock(optionMutex_);
  char v[64];
  fprintf(stderr, "OPENMP DISPLAY ENVIRONMENT BEGIN\n");
  fprintf(stderr, "  %-22s = '%s'\n", "_OPENMP", "pomp (onnx-mlir)");
  int lim = limit();
  snprintf(v, sizeof(v), "%d", lim);
  fprintf(stderr, "  %-22s = %-12s [%s]\n", "OMP_THREAD_LIMIT", v,
      limitFromEnv_ ? "environment"
                    : "pomp_set_thread_limit or pomp_set_option");
  // Team sizes as used: clamped to the limit.
  snprintf(v, sizeof(v), "'%d,%d'", numThreads(0), numThreads(1));
  fprintf(stderr, "  %-22s = %-12s [%s]\n", "OMP_NUM_THREADS", v,
      sourceName(sources_[0]));
  snprintf(v, sizeof(v), "'%d'", maxActive());
  fprintf(stderr, "  %-22s = %-12s [%s%s]\n", "OMP_MAX_ACTIVE_LEVELS", v,
      sourceName(sources_[1]),
      sources_[1] == kDefault && nthreadsNested_ ? ", from OMP_NUM_THREADS"
                                                 : "");
  snprintf(v, sizeof(v), "'%d'", treeMin());
  fprintf(stderr, "  %-22s = %-12s [%s]\n", "POMP_TREE_BARRIER", v,
      sourceName(sources_[2]));
  if (spinAdaptive())
    snprintf(v, sizeof(v), "'auto:%ld'", spinMaxUs());
  else
    snprintf(v, sizeof(v), "'%ld'", spinUs());
  fprintf(stderr, "  %-22s = %-12s [%s]\n", "POMP_SPIN", v,
      sourceName(sources_[3]));
  fprintf(stderr, "  %-22s = '%s'\n", "POMP_IGNORE_ENV",
      ignoreEnv_ ? "TRUE" : "FALSE");
  if (displayEnv_ == 2) {
    fprintf(stderr, "  %-22s = '%d'\n", "[pomp] max threads", kMaxThreads);
    fprintf(stderr, "  %-22s = '%d'\n", "[pomp] cache line", kCacheLine);
    fprintf(stderr, "  %-22s = '%ld-%ld'\n", "[pomp] auto spin, us",
        kSpinFloorUs, spinMaxUs());
#ifdef POMP_USE_PTHREAD_KEY
    fprintf(stderr, "  %-22s = '%s'\n", "[pomp] thread slot", "pthread key");
#else
    fprintf(stderr, "  %-22s = '%s'\n", "[pomp] thread slot", "__thread");
#endif
#ifdef POMP_STATS
    fprintf(stderr, "  %-22s = '%s'\n", "[pomp] counters", "on");
#endif
  }
  fprintf(stderr, "OPENMP DISPLAY ENVIRONMENT END\n");
}

// "n" or "n1,n2"; missing entries repeat the last one. Clamped to the limit
// once it is known (setLimit), or now if it is.
bool Settings::setNumThreads(const char *s, bool apply) {
  int vals[kMaxActive] = {};
  int n = 0;
  const char *end;
  while (n < kMaxActive) {
    long v = parseInt(s, &end);
    if (v < 1)
      return false;
    vals[n++] = v > kMaxThreads ? kMaxThreads : int(v);
    if (*end != ',')
      break;
    s = end + 1;
  }
  if (*end && *end != ',')
    return false;
  for (int i = n; i < kMaxActive; ++i)
    vals[i] = vals[i - 1];
  if (!apply)
    return true;
  // Under limitMutex_ too: setLimit reads envNthreads_.
  ScopedLock<Mutex> lock(limitMutex_);
  for (int i = 0; i < kMaxActive; ++i)
    envNthreads_[i] = vals[i];
  // As libomp: a list of more than one level implies nesting, unless
  // OMP_MAX_ACTIVE_LEVELS is set.
  nthreadsNested_ = n > 1;
  if (sources_[1] == kDefault)
    maxActive_.store(nthreadsNested_ ? kMaxActive : 1, Order::Relaxed);
  int lim = limit();
  if (lim)
    for (int i = 0; i < kMaxActive; ++i)
      nthreads_[i].store(vals[i] < lim ? vals[i] : lim);
  return true;
}

// 1 or 2; 0 means 1 and larger values mean 2.
bool Settings::setMaxActive(const char *s, bool apply) {
  const char *end;
  long v = parseInt(s, &end);
  if (v < 0 || *end)
    return false;
  if (apply)
    maxActive_.store(v < 1 ? 1 : (v > kMaxActive ? kMaxActive : int(v)),
      Order::Relaxed);
  return true;
}

bool Settings::setTreeMin(const char *s, bool apply) {
  const char *end;
  long v = parseInt(s, &end);
  if (v < 0 || *end)
    return false;
  if (apply)
    treeMin_.store(int(v), Order::Relaxed);
  return true;
}

// auto, auto:MAX, or a fixed number of us.
bool Settings::setSpin(const char *s, bool apply) {
  const char *end;
  long v;
  while (*s == ' ' || *s == '\t')
    ++s;
  if (!strncmp(s, "auto", 4)) {
    s += 4;
    long maxUs = kDefaultSpinMaxUs;
    if (*s == ':') {
      v = parseInt(s + 1, &end);
      if (v < 0 || *end)
        return false;
      maxUs = v;
    } else if (*s) {
      return false;
    }
    if (apply) {
      spinMaxUs_.store(maxUs, Order::Relaxed);
      spinUs_.store(-1, Order::Relaxed);
    }
    return true;
  }
  v = parseInt(s, &end);
  if (v < 0 || *end)
    return false;
  if (apply)
    spinUs_.store(v, Order::Relaxed);
  return true;
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
  // Not yet visible to other threads: the limit_ store below publishes the
  // descriptors.
  threads = t;
  teams = m;
  for (int32_t i = 0; i < limit; ++i)
    t[i].setSlot(i);
  for (int32_t i = 0; i < limit; ++i)
    m[i].init(i, t[i]);
  m[limit].init(limit, t[0]);
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
