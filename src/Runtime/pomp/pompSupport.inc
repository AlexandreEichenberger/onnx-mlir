/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- pompSupport.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Generic support code of pomp.cpp: errors, spinning, atomics, pthread locks,
// and the call of an outlined function. Included in pomp.cpp's single
// translation unit, so all of it stays visible to the inliner; same language
// rules as pomp.cpp. Uses kMaxArgs, defined before the include.
//
//===----------------------------------------------------------------------===//

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Globals must be constant-initialized: a load-time constructor would need the
// C++ runtime. Checked by C++20 constinit, or clang's attribute in C++17.
#if defined(__cpp_constinit)
#define POMP_CONSTINIT constinit
#elif defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::require_constant_initialization)
#define POMP_CONSTINIT [[clang::require_constant_initialization]]
#endif
#endif
#ifndef POMP_CONSTINIT
#define POMP_CONSTINIT
#endif

namespace {

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

[[noreturn]] void fatal(const char *msg, const char *detail = "") {
  fprintf(stderr, "pomp: fatal: %s%s\n", msg, detail);
  fflush(stderr);
  abort();
}

// Abort if a pthread call fails.
inline void checkPthread(int rc, const char *call) {
  if (rc != 0)
    fatal("pthread call failed: ", call);
}

// A monotonic clock in microseconds, cheap enough to read every few dozen
// spins: STCKF on Linux on Z, the vDSO's clock_gettime elsewhere.
inline uint64_t nowUs() {
#if defined(__s390x__) && defined(__linux__)
  uint64_t t;
  __asm__ volatile("stckf %0" : "=Q"(t) : : "cc");
  return t >> 12; // Bit 51 of the TOD clock is 1 us.
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000u + uint64_t(ts.tv_nsec) / 1000u;
#endif
}

// The spin of a wait: more() is true until `limitUs` microseconds after its
// first clock read, which comes kSpinClockStride calls in, so that waits
// shorter than that never read the clock. A limit of 0 never spins.
class SpinBudget {
public:
  explicit SpinBudget(long limitUs) : limitUs_(uint64_t(limitUs)) {}
  bool more() {
    if (limitUs_ == 0)
      return false;
    if (++n_ % kSpinClockStride)
      return true;
    uint64_t t = nowUs();
    if (n_ == kSpinClockStride)
      start_ = t;
    return t - start_ < limitUs_;
  }

private:
  uint64_t limitUs_;
  uint64_t start_ = 0;
  long n_ = 0;
};

// The spin of one kind of wait of one thread, under POMP_SPIN=auto:
// covers twice the last short wait, so that waits like it do not sleep, and
// halves after each wait longer than the ceiling, down to the floor.
class AdaptiveSpin {
public:
  // The spin for the next wait; starts at the ceiling.
  // POMP_SPIN may lower the ceiling at any time: never above it.
  long next(long maxUs) const { return us_ && us_ < maxUs ? us_ : maxUs; }
  // After a wait of `waitedUs`, from its start to its end.
  void update(uint64_t waitedUs, long maxUs) {
    long cur = next(maxUs);
    long floor = kSpinFloorUs < maxUs ? kSpinFloorUs : maxUs;
    if (waitedUs <= uint64_t(maxUs)) {
      long want = long(2 * waitedUs);
      cur = want > cur ? want : cur;
      cur = cur < maxUs ? cur : maxUs;
    } else {
      cur = cur / 2 > floor ? cur / 2 : floor;
    }
    // 0 means "not started", so store at least 1 (a ceiling of 0 never
    // spins: SpinBudget(0) ends at once).
    us_ = cur > 0 ? cur : 1;
  }

private:
  long us_ = 0; // 0: not started.
};

inline void relax() {
#if defined(__x86_64__) || defined(__i386__)
  __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
  __asm__ volatile("yield" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

// Parse a positive integer; return -1 if the string is not one.
long parseInt(const char *s, const char **end) {
  char *e;
  while (*s == ' ' || *s == '\t')
    ++s;
  if (*s < '0' || *s > '9')
    return -1;
  long v = strtol(s, &e, 10);
  while (*e == ' ' || *e == '\t')
    ++e;
  if (end)
    *end = e;
  return v;
}

// -----------------------------------------------------------------------------
// Atomics and locks
//
// Atomics wrap the __atomic builtins (C11 memory model, no header); seq_cst
// unless told otherwise. Locks wrap pthread's.
// -----------------------------------------------------------------------------

enum class Order : int {
  Relaxed = __ATOMIC_RELAXED,
  Acquire = __ATOMIC_ACQUIRE,
  Release = __ATOMIC_RELEASE,
  AcqRel = __ATOMIC_ACQ_REL,
  SeqCst = __ATOMIC_SEQ_CST,
};

inline void fence(Order o) { __atomic_thread_fence(int(o)); }

// An atomic integer or pointer.
template <class T>
class Atomic {
public:
  constexpr Atomic() : value_() {}
  constexpr explicit Atomic(T v) : value_(v) {}
  Atomic(const Atomic &) = delete;
  Atomic &operator=(const Atomic &) = delete;

  T load(Order o = Order::SeqCst) const {
    return __atomic_load_n(&value_, int(o));
  }
  void store(T v, Order o = Order::SeqCst) {
    __atomic_store_n(&value_, v, int(o));
  }
  T exchange(T v, Order o = Order::SeqCst) {
    return __atomic_exchange_n(&value_, v, int(o));
  }
  T fetchAdd(T v, Order o = Order::SeqCst) {
    return __atomic_fetch_add(&value_, v, int(o));
  }
  T fetchSub(T v, Order o = Order::SeqCst) {
    return __atomic_fetch_sub(&value_, v, int(o));
  }
  T fetchAnd(T v, Order o = Order::SeqCst) {
    return __atomic_fetch_and(&value_, v, int(o));
  }
  T fetchOr(T v, Order o = Order::SeqCst) {
    return __atomic_fetch_or(&value_, v, int(o));
  }
  bool compareExchange(T &expected, T desired, Order success = Order::SeqCst,
      Order failure = Order::SeqCst) {
    return __atomic_compare_exchange_n(
        &value_, &expected, desired, false, int(success), int(failure));
  }
  bool compareExchangeWeak(T &expected, T desired,
      Order success = Order::SeqCst, Order failure = Order::SeqCst) {
    return __atomic_compare_exchange_n(
        &value_, &expected, desired, true, int(success), int(failure));
  }

private:
  T value_;
};

// A pthread mutex, zeroed until init(), which must come before any use.
class Mutex {
public:
  constexpr Mutex() : mutex_() {}
  Mutex(const Mutex &) = delete;
  Mutex &operator=(const Mutex &) = delete;

  void init() {
    checkPthread(pthread_mutex_init(&mutex_, nullptr), "pthread_mutex_init");
  }
  void lock() {
    checkPthread(pthread_mutex_lock(&mutex_), "pthread_mutex_lock");
  }
  void unlock() {
    checkPthread(pthread_mutex_unlock(&mutex_), "pthread_mutex_unlock");
  }
  pthread_mutex_t *native() { return &mutex_; }

private:
  pthread_mutex_t mutex_;
};

// A pthread condition, zeroed until init().
class Condition {
public:
  constexpr Condition() : cond_() {}
  Condition(const Condition &) = delete;
  Condition &operator=(const Condition &) = delete;

  void init() {
    checkPthread(pthread_cond_init(&cond_, nullptr), "pthread_cond_init");
  }
  void signal() {
    checkPthread(pthread_cond_signal(&cond_), "pthread_cond_signal");
  }
  void wait(Mutex &m) {
    checkPthread(pthread_cond_wait(&cond_, m.native()), "pthread_cond_wait");
  }

private:
  pthread_cond_t cond_;
};

// Holds a lock for the lifetime of the object.
template <class Lockable>
class ScopedLock {
public:
  explicit ScopedLock(Lockable &l) : lockable_(l) { lockable_.lock(); }
  ~ScopedLock() { lockable_.unlock(); }
  ScopedLock(const ScopedLock &) = delete;
  ScopedLock &operator=(const ScopedLock &) = delete;

private:
  Lockable &lockable_;
};

// Counters of sleeps and wake-ups, built with -DPOMP_STATS only (read by
// pomp_get_stats). Relaxed increments of shared counters: for diagnosis, not
// for timing runs.
enum StatKind {
  kStatRegions,    // Forks of a team of more than one thread.
  kStatIdleSleeps, // An idle worker went to sleep, waiting for work.
  kStatTeamSleeps, // A join or barrier wait went to sleep.
  kStatPosts,      // Posts, by any waker.
  kStatForkPosts,  // Posts by a leader waking its workers at a fork.
  kStatPostNs,     // Time inside post(), in ns.
  kStatSleepNs,    // Time asleep (waitPost), in ns.
  kStats
};
#ifdef POMP_STATS
inline uint64_t nowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000u + uint64_t(ts.tv_nsec);
}
POMP_CONSTINIT uint64_t statCounters[kStats] = {};
inline void stat(StatKind k, uint64_t n = 1) {
  __atomic_fetch_add(&statCounters[k], n, __ATOMIC_RELAXED);
}
#define POMP_STAT(k) stat(k)
#define POMP_STAT_TIME(k, stmt)                                                \
  do {                                                                         \
    uint64_t t0_ = nowNs();                                                    \
    stmt;                                                                      \
    stat(k, nowNs() - t0_);                                                    \
  } while (0)
#else
#define POMP_STAT(k) ((void)0)
#define POMP_STAT_TIME(k, stmt) stmt
#endif

// A binary flag to sleep on. A post only means "re-check your condition".
class Post {
public:
  void init() {
    mutex_.init();
    cond_.init();
  }
  // Set the flag and wake the waiter.
  void post() {
    ScopedLock<Mutex> lock(mutex_);
    posted_ = true;
    cond_.signal();
  }
  // Wait until the flag is set, then consume it.
  void wait() {
    ScopedLock<Mutex> lock(mutex_);
    while (!posted_)
      cond_.wait(mutex_);
    posted_ = false;
  }
  // Drop a pending post.
  void clear() {
    ScopedLock<Mutex> lock(mutex_);
    posted_ = false;
  }

private:
  Mutex mutex_;
  Condition cond_;
  bool posted_ = false;
};

// -----------------------------------------------------------------------------
// Calling an outlined function
//
// An outlined function is declared variadic but defined with fixed
// parameters (two int32_t *, then one void * per argument). Call it through
// its exact type, as libomp's portable invoke does, since variadic and fixed
// calls differ on some ABIs: a compile-time table holds one caller per count.
// -----------------------------------------------------------------------------

using Microtask = void (*)(int32_t *gtid, int32_t *btid, ...);

// call() passes argv[I]... to an outlined function.
template <unsigned... I>
struct Args {
  template <unsigned>
  using Ptr = void *;

  static void call(Microtask fn, int32_t *gtid, int32_t *tid, void **argv) {
    using Exact = void (*)(int32_t *, int32_t *, Ptr<I>...);
    reinterpret_cast<Exact>(fn)(gtid, tid, argv[I]...);
  }
};

// FirstArgs<N>::Type is Args<0, ..., N-1>.
template <unsigned N, unsigned... I>
struct FirstArgs : FirstArgs<N - 1, N - 1, I...> {};
template <unsigned... I>
struct FirstArgs<0, I...> {
  using Type = Args<I...>;
};

// For Args<0, ..., N-1>: callers[n] calls with n arguments, for n < N.
template <class>
struct CallTable;
template <unsigned... N>
struct CallTable<Args<N...>> {
  using Caller = void (*)(Microtask, int32_t *, int32_t *, void **);
  static constexpr Caller callers[] = {&FirstArgs<N>::Type::call...};
};

// Call `fn` with the gtid and tid pointers it takes, and argv[0..argc-1].
void invoke(
    Microtask fn, int32_t gtid, int32_t tid, int32_t argc, void **argv) {
  using Table = CallTable<FirstArgs<kMaxArgs + 1>::Type>;
  if (argc < 0)
    fatal("negative argument count for an outlined function");
  if (argc > kMaxArgs)
    fatal("too many arguments to an outlined function");
  Table::callers[argc](fn, &gtid, &tid, argv);
}

} // namespace
