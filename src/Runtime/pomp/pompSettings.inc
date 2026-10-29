/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pompSettings.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the settings, read from the environment once and set by
// pomp_set_option, and the thread limit.
//
//===----------------------------------------------------------------------===//

// Settings: defaults, then the environment, then pomp_set_option calls. The
// environment is read at init (the first call of any entry point) and gives
// initial values; as in OpenMP, a call overrides them. The exception is the
// thread limit: OMP_THREAD_LIMIT wins over calls. See "Environment" in
// POMP.md.
class Settings {
public:
  // Where a setting's value comes from, in the order they apply.
  enum Source : int8_t { kDefault, kEnv, kCall };
  // pomp_set_option: set setting `name` (its environment variable's name)
  // from `value`. POMP_ERR_LIMIT: unknown name or invalid value;
  // POMP_ERR_STATE: the setting cannot change now.
  int setOption(const char *name, const char *value, Source from);
  // At each successful open: from the first one on, only POMP_SPIN may
  // change; and OMP_DISPLAY_ENV prints the settings, once.
  void freeze();
  // Read the environment, once, at init.
  void init();
  // Fix the limit, once: OMP_THREAD_LIMIT if set (at init), else the first
  // open's fallback, once its descriptors are allocated (Pool::open).
  void setLimit(int lim);

  // The thread limit, OpenMP's thread-limit-var, slot 0 (a team's root)
  // included: OMP_THREAD_LIMIT if set, else the fallback of the first
  // successful open. 0 while neither is known; never changes once set.
  int limit() const { return limit_.load(Order::Relaxed); }
  // Slots 0..limit-1.
  SlotMask limitSlots() const { return limitSlots_.load(Order::Relaxed); }
  // Warn once that no limit is known.
  void warnNoLimit();
  // Default team size of a fork at `active` enclosing active levels.
  int numThreads(int active) const {
    return nthreads_[active < kMaxActive ? active : kMaxActive - 1].load(
        Order::Relaxed);
  }
  int maxActive() const { return maxActive_.load(Order::Relaxed); }
  // POMP_SPIN: a fixed spin in us, or -1 for auto, which adapts up to
  // spinMaxUs().
  bool spinAdaptive() const { return spinUs_.load(Order::Relaxed) < 0; }
  long spinUs() const { return spinUs_.load(Order::Relaxed); }
  long spinMaxUs() const { return spinMaxUs_.load(Order::Relaxed); }
  int treeMin() const { return treeMin_.load(Order::Relaxed); }

private:
  // Parsers: return false if `s` is invalid (the setting is unchanged);
  // otherwise set the setting. Under optionMutex_.
  bool setNumThreads(const char *s);
  bool setMaxActive(const char *s);
  bool setTreeMin(const char *s);
  bool setSpin(const char *s);
  // Settings with a source, indexed as kOptions below.
  static constexpr int kNumOptions = 4;

  Atomic<int> frozen_; // Set by the first open (freeze).
  // Serializes setOption (calls, and init's reads of the environment).
  Mutex optionMutex_;
  Source sources_[kNumOptions] = {};
  int displayEnv_ = 0; // OMP_DISPLAY_ENV: 0, 1 (true), 2 (verbose).
  Atomic<int> displayed_;
  int envNthreads_[kMaxActive] = {}; // OMP_NUM_THREADS as set, 0: unset.
  // Whether OMP_NUM_THREADS listed more than one level, for the default of
  // OMP_MAX_ACTIVE_LEVELS.
  bool nthreadsNested_ = false;
  Atomic<int> maxActive_;
  // The limit (see limit()); written under limitMutex_.
  Mutex limitMutex_;
  Atomic<int> limit_;
  AtomicMask limitSlots_;            // Slots 0..limit_-1.
  Atomic<int> nthreads_[kMaxActive]; // Default team sizes, clamped to limit_.
  bool limitFromEnv_ = false;        // For OMP_DISPLAY_ENV; set at init.
  Atomic<int> noLimitSeen_;
  // Read at every wait: atomics, as POMP_SPIN may change while regions run.
  Atomic<long> spinUs_;
  Atomic<long> spinMaxUs_;
  Atomic<int> treeMin_; // Team size from which the tree barrier is used.
};

POMP_CONSTINIT Settings settings;

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

void Settings::init() {
  limitMutex_.init();
  optionMutex_.init();

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
  // the first open (see Pool::open).
  if ((s = getenv("OMP_THREAD_LIMIT"))) {
    const char *end;
    long v = parseInt(s, &end);
    if (*end || v < 1 || v > kMaxThreads)
      fprintf(stderr,
          "pomp: warning: OMP_THREAD_LIMIT=\"%s\" is not in [1, %d], and is "
          "ignored; pomp_open_pool must give the limit\n",
          s, kMaxThreads);
    else {
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
  if (from == kCall && !kOptions[k].anytime && frozen_.load())
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

// OMP_DISPLAY_ENV: in OpenMP's format, each line followed by where the value
// comes from.
void Settings::freeze() {
  frozen_.store(1);
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

void Settings::setLimit(int lim) {
  ScopedLock<Mutex> lock(limitMutex_);
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
