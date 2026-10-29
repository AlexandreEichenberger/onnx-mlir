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
// thread limit, with the allocation of the descriptors at the first open.
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

// Called on each entry as this slot, before it is recruitable, and at the
// allocation of a root block. Entries as a slot are serialized by `present_`.
// The post is initialized once per descriptor, never again: a waker of a
// previous occupant (or owner, for a root block) may still be inside its
// post() ("Root blocks", rule 2, in pompRegion.inc).
inline void Thread::init() {
  if (!initialized_) {
    post_.init();
    initialized_ = true;
  }
}

// Concurrent first calls wait for the one that does it.
// Initialized values: 0: no, 1: in progress, 2: done.
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
  pool.init();
  Self::createStorage();
  roots.init();

  // Defaults; then each setting the environment sets.
  maxActive_.store(1, Order::Relaxed);
  treeMin_.store(kDefaultTreeBarrier, Order::Relaxed);
  spinUs_.store(-1, Order::Relaxed);
  spinMaxUs_.store(kDefaultSpinMaxUs, Order::Relaxed);
  // OMP_DISPLAY_ENV: TRUE or VERBOSE (any case), as in OpenMP.
  const char *s = getenv("OMP_DISPLAY_ENV");
  if (s && (!strcasecmp(s, "true") || !strcmp(s, "1")))
    displayEnv_ = 1;
  else if (s && !strcasecmp(s, "verbose"))
    displayEnv_ = 2;
  // OMP_NUM_THREADS first: it gives OMP_MAX_ACTIVE_LEVELS its default.
  static const char *const names[] = {"OMP_NUM_THREADS",
      "OMP_MAX_ACTIVE_LEVELS", "POMP_TREE_BARRIER", "POMP_SPIN"};
  for (const char *name : names)
    if ((s = getenv(name)) && setOption(name, s, kEnv) != POMP_OK)
      fprintf(
          stderr, "pomp: warning: %s=\"%s\" is invalid and ignored\n", name, s);

  // OMP_THREAD_LIMIT, if valid, is the limit; nothing is allocated before
  // the first open (see openLimit).
  if ((s = getenv("OMP_THREAD_LIMIT"))) {
    const char *end;
    long v = parseInt(s, &end);
    if (*end || v < 1 || v > kMaxThreads)
      fprintf(stderr,
          "pomp: warning: OMP_THREAD_LIMIT=\"%s\" is not in [1, %d], and is "
          "ignored; pomp_open_pool must give the limit\n",
          s, kMaxThreads);
    else {
      ScopedLock<Mutex> lock(limitMutex_);
      limitFromEnv_ = true;
      setLimit(int(v));
    }
  }
}

// The settings pomp_set_option knows, in the order of sources_. Whether each
// may change once the pool has opened: team sizes and the barrier kind are
// read at forks, so only before; POMP_SPIN at every wait, so at any time.
namespace {
struct OptionDesc {
  const char *name;
  bool anytime;
  bool (Settings::*set)(const char *);
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
  int k = 0;
  while (k < kNumOptions && strcmp(name, kOptions[k].name))
    ++k;
  if (k == kNumOptions)
    return POMP_ERR_LIMIT;
  if (from == kCall && !kOptions[k].anytime && pool.wasOpened())
    return POMP_ERR_STATE;
  // As in OpenMP, the environment gives the initial value (init reads it
  // before any call is served) and a call overrides it.
  ScopedLock<Mutex> lock(optionMutex_);
  if (!(this->*kOptions[k].set)(value))
    return POMP_ERR_LIMIT;
  sources_[k] = from;
  return POMP_OK;
}

namespace {
const char *sourceName(Settings::Source s) {
  return s == Settings::kEnv    ? "environment"
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
  snprintf(v, sizeof(v), "'%d'", limit());
  fprintf(stderr, "  %-22s = %-12s [%s]\n", "OMP_THREAD_LIMIT", v,
      limitFromEnv_ ? "environment" : "pomp_open_pool fallback");
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
bool Settings::setNumThreads(const char *s) {
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
bool Settings::setMaxActive(const char *s) {
  const char *end;
  long v = parseInt(s, &end);
  if (v < 0 || *end)
    return false;
  maxActive_.store(
      v < 1 ? 1 : (v > kMaxActive ? kMaxActive : int(v)), Order::Relaxed);
  return true;
}

bool Settings::setTreeMin(const char *s) {
  const char *end;
  long v = parseInt(s, &end);
  if (v < 0 || *end)
    return false;
  treeMin_.store(int(v), Order::Relaxed);
  return true;
}

// auto, auto:MAX, or a fixed number of us.
bool Settings::setSpin(const char *s) {
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
    spinMaxUs_.store(maxUs, Order::Relaxed);
    spinUs_.store(-1, Order::Relaxed);
    return true;
  }
  v = parseInt(s, &end);
  if (v < 0 || *end)
    return false;
  spinUs_.store(v, Order::Relaxed);
  return true;
}

int Settings::openLimit(int fallback) {
  ScopedLock<Mutex> lock(limitMutex_);
  if (allocated_.load(Order::Relaxed))
    return POMP_OK;
  // The first open: allocate for OMP_THREAD_LIMIT, else for `fallback`, and
  // only then fix the limit. On failure nothing is fixed, so a later open may
  // retry, with another fallback.
  int lim = limit() ? limit() : fallback;
  if (!lim)
    return POMP_ERR_LIMIT;
  if (!allocate(lim))
    return POMP_ERR_NOMEM;
  if (!limit())
    setLimit(lim);
  return POMP_OK;
}

bool Settings::allocate(int n) {
  // One block, never freed: `n` threads, then `n` teams from the next line,
  // for the pool threads; index 0 is unused (roots have their own).
  size_t tBytes =
      (size_t(n) * sizeof(Thread) + kCacheLine - 1) & ~size_t(kCacheLine - 1);
  char *p =
      static_cast<char *>(allocLines(1, tBytes + size_t(n) * sizeof(Team)));
  if (!p)
    return false;
  Thread *t = reinterpret_cast<Thread *>(p);
  Team *m = reinterpret_cast<Team *>(p + tBytes);
  // Not yet visible to other threads: the allocated_ store below publishes
  // the descriptors. Pool threads init their sleep on entry.
  threads = t;
  teams = m;
  for (int32_t i = 1; i < n; ++i) {
    t[i].initIdentity(i, nullptr, &m[i]);
    m[i].init(i, t[i]);
  }
  // Publishes the descriptors (see allocatedAcquire).
  allocated_.store(1, Order::Release);
  return true;
}

void Settings::setLimit(int lim) {
  for (int i = 0; i < kMaxActive; ++i) {
    int n = envNthreads_[i];
    nthreads_[i].store(n && n < lim ? n : lim);
  }
  limitSlots_.store(SlotMask::below(lim));
  limit_.store(lim);
}

void Settings::warnNoLimit() {
  if (!noLimitSeen_.exchange(1))
    fprintf(stderr,
        "pomp: warning: omp_get_max_threads/omp_get_thread_limit return 1: "
        "no thread limit is known yet (OMP_THREAD_LIMIT is unset or invalid, "
        "and no pomp_open_pool has given one); until one does, parallel "
        "regions run on one thread\n");
}
