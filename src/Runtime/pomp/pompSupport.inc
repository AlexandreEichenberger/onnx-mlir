/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- pompSupport.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Support code of pomp.cpp that is not specific to pomp: constant
// initialization, errors, spinning, atomics, pthread locks and conditions,
// and the call of an outlined function. Included by pomp.cpp only, in its
// single translation unit, so that every function here stays visible to the
// inliner; it is not a separate source. The same rules as pomp.cpp apply: no
// C++ header, no C++ runtime, constant initialization only (see "Language" in
// POMP.md). pomp.cpp defines kMaxArgs, the most arguments invoke() passes,
// before including it.
//
//===----------------------------------------------------------------------===//

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Every global must be initialized at compile time: a constructor run at load
// would need the C++ runtime's static-init machinery. C++20 checks it with
// constinit, and clang in C++17 with its own attribute.
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

// Abort if a pthread call fails: pomp has no way to continue without it.
inline void checkPthread(int rc, const char *call) {
  if (rc != 0)
    fatal("pthread call failed: ", call);
}

inline void relax() {
#if defined(__x86_64__) || defined(__i386__)
  __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
  __asm__ volatile("yield" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

// One iteration of a spin loop that also yields the CPU periodically: relax,
// and every `every` iterations call sched_yield, which lets another thread
// that shares the CPU (or its core, with SMT) run, and the scheduler move
// this one. `every` 0 never yields. The countdown is a local, so that a
// spin costs one decrement and one compare more than relax().
class SpinYield {
public:
  explicit SpinYield(long every) : every_(every), left_(every) {}
  void spin() {
    relax();
    if (left_ > 0 && --left_ == 0) {
      sched_yield();
      left_ = every_;
    }
  }

private:
  long every_;
  long left_;
};

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
// Atomics use the compiler's __atomic builtins, on which both <stdatomic.h>
// and <atomic> are built: the same instructions and the same memory model as
// C11, without a header. As in C11, an operation is sequentially consistent
// unless told otherwise. Locks are pthread's, behind small classes.
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

// A pthread mutex. Its storage starts zeroed, so that static arrays of them
// stay in .bss; init() must run before any other use.
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

// A pthread condition, zeroed until init() as a Mutex.
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

// A binary "posted" flag with its mutex and condition, to sleep on. A post
// only means "re-check": the waiter tests its own condition again.
class Post {
public:
  void init() {
    mutex_.init();
    cond_.init();
  }
  void post() {
    ScopedLock<Mutex> lock(mutex_);
    posted_ = true;
    cond_.signal();
  }
  void wait() {
    ScopedLock<Mutex> lock(mutex_);
    while (!posted_)
      cond_.wait(mutex_);
    posted_ = false;
  }
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
// The compiler declares an outlined function variadic, but defines it with
// fixed parameters: the two int32_t pointers, then one void * per shared
// argument. Call each through its exact type, as libomp's portable
// kmp_invoke_microtask.cpp does, so that the call is valid on ABIs where
// variadic and fixed calls differ. A table, built at compile time, holds one
// caller per argument count.
// -----------------------------------------------------------------------------

using Microtask = void (*)(int32_t *gtid, int32_t *btid, ...);

// The argument indices I...: call() passes argv[I]... to an outlined function.
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
void invoke(Microtask fn, int32_t gtid, int32_t tid, int argc, void **argv) {
  using Table = CallTable<FirstArgs<kMaxArgs + 1>::Type>;
  if (argc < 0 || argc > kMaxArgs)
    fatal("too many arguments to an outlined function");
  Table::callers[argc](fn, &gtid, &tid, argv);
}

} // namespace
