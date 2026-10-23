/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pomp.cpp ----------------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// pomp: a minimal pool-based OpenMP runtime. See POMP.md for the design.
//
// Implements the libomp entry points that onnx-mlir models (MLIR's
// OpenMPIRBuilder) and zDNNx (clang) emit:
//   __kmpc_global_thread_num   __kmpc_fork_call __kmpc_push_num_threads
//   __kmpc_push_proc_bind      __kmpc_serialized_parallel
//   __kmpc_end_serialized_parallel                       __kmpc_barrier
//   __kmpc_for_static_init_{4,4u,8,8u}                   __kmpc_for_static_fini
// plus a few omp_* queries.
//
// Model: one root (the initial thread, slot 0) and up to 63 worker threads
// lent by the host through the pool (slots 1..limit-1). One 64-bit word,
// `busy`, holds one bit per slot; a bit is clear iff that slot is an idle
// worker in the pool. A leader recruits a team with a single CAS on `busy`,
// hands each worker the work through its descriptor, wakes it by bumping its
// private epoch, and clears the team's bits again after the join. At most two
// levels are active; deeper or empty forks run inline as a team of one.
//
// The join polls one done flag per member, and barriers of more than four
// threads use a 4-ary tree; see POMP.md for the measurements behind both.
// Every wait, idle or in a team, spins and then sleeps.
//
// Release does not block the leader: it closes the pool, claims the idle
// workers and marks them in `released`. Spinning workers see their bit and
// leave by themselves; each leaving worker wakes up to two sleeping ones, so
// the wake-ups spread as a tree. See POMP.md for the scenarios supported.
//
// Language: C++ for its classes and templates only. No virtual functions,
// exceptions, RTTI, dynamic allocation, nor any C++ header; every global is
// constant-initialized and trivially destructible. The object file thus needs
// only the C library and pthreads, as a C one does, and links into C
// programs with the C driver. See "Language" in POMP.md.
//
//===----------------------------------------------------------------------===//

#include "pomp.h"

#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
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
// Configuration
// -----------------------------------------------------------------------------

constexpr int kMaxThreads = POMP_MAX_THREADS;
static_assert(kMaxThreads <= 64, "slots are bits of one 64-bit word");
// Teams: one led by each slot at level 2, and the root's at level 1.
constexpr int kTeams = kMaxThreads + 1;

#if defined(__s390x__) || defined(__MVS__)
constexpr int kCacheLine = 256;
#elif defined(__aarch64__) && defined(__APPLE__)
constexpr int kCacheLine = 128;
#else
constexpr int kCacheLine = 64;
#endif

// Max shared args passed to an outlined fn: one more than the 15 of libomp's
// portable kmp_invoke_microtask.cpp.
constexpr int kMaxArgs = 16;
// Max nesting levels recorded; deeper ones are serialized and only counted.
constexpr int kMaxDepth = 8;
constexpr int kMaxActive = 2; // Max active (more than 1 thread) levels.
// Spin iterations of any wait (idle, join, barrier) before it sleeps.
constexpr long kDefaultSpin = 200000;
constexpr long kYieldSpin = 4096; // Team-wait spin iterations before yield.

// Teams of at least this many threads use the tree barrier, smaller ones the
// central one (POMP_TREE_BARRIER overrides it). Measured: on z17 the tree is
// 41-70% faster from 16 threads on and on par at 8; on Apple M-series the
// central barrier is faster up to 16 threads. See POMP.md.
#if defined(__s390x__) || defined(__MVS__)
constexpr int kDefaultTreeBarrier = 5;
#else
constexpr int kDefaultTreeBarrier = kMaxThreads + 1; // Never.
#endif
constexpr int kTreeArity = 4;

// Schedule kinds (values of libomp's enum sched_type).
constexpr int32_t kSchStaticChunked = 33;
constexpr int32_t kSchStatic = 34;
constexpr int32_t kSchModifierMask = (1 << 29) | (1 << 30);

using Microtask = void (*)(int32_t *gtid, int32_t *btid, ...);

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

constexpr uint64_t bit(int slot) { return uint64_t(1) << slot; }

inline int popcount(uint64_t x) { return __builtin_popcountll(x); }

// Index of the lowest set bit of a nonzero x.
inline int lowestSlot(uint64_t x) { return __builtin_ctzll(x); }

// Mask of the n lowest bits.
constexpr uint64_t bitsBelow(int n) {
  return n >= 64 ? UINT64_MAX : bit(n) - 1;
}

// Keep the n lowest set bits of x.
inline uint64_t lowestBits(uint64_t x, int n) {
  uint64_t r = 0;
  for (; n > 0 && x; --n) {
    uint64_t low = x & (~x + 1);
    r |= low;
    x &= ~low;
  }
  return r;
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

// Sleep: a binary "posted" flag with its mutex and condition. A post only
// means "re-check"; see "Wake-up and sleep".
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
// Thread identity, one per-thread word: the slot of the calling thread (0 for
// any non-worker), plus a root-owner flag while a non-worker runs regions as
// slot 0. Every non-worker reads slot 0, so the flag is what tells the root of
// the running regions from another host thread.
//
// The word also holds what belongs to the thread rather than to its slot,
// which other non-workers share: the team size pushed by
// __kmpc_push_num_threads for its next fork, and the depth of the regions it
// runs inline while another thread owns slot 0 ("foreign" regions).
// -----------------------------------------------------------------------------

class SelfWord {
public:
  constexpr explicit SelfWord(int32_t word = 0) : word_(word) {}

  // The calling thread's word, and its update.
  static SelfWord current();
  static void set(SelfWord w);
  // Create the per-thread storage, once, before any thread has a slot.
  static void createStorage();

  int32_t slot() const { return word_ & kSlotBits; }
  bool ownsRoot() const { return (word_ & kRootOwner) != 0; }
  // A worker, or the owner of slot 0: the thread has a descriptor.
  bool hasDescriptor() const { return (word_ & (kSlotBits | kRootOwner)) != 0; }
  // A worker, or a thread inside a region.
  bool inPomp() const { return (word_ & kInPompBits) != 0; }
  int32_t pending() const { return (word_ & kPendingBits) >> kPendingShift; }
  bool inForeign() const { return (word_ & kForeignBits) != 0; }
  int32_t foreignDepth() const { return (word_ & kForeignBits) / kForeignOne; }

  SelfWord withPending(int32_t nth) const {
    return SelfWord((word_ & ~kPendingBits) | (nth << kPendingShift));
  }
  SelfWord withoutPending() const { return SelfWord(word_ & ~kPendingBits); }
  SelfWord withRootOwner() const { return SelfWord(word_ | kRootOwner); }
  SelfWord withoutRootOwner() const { return SelfWord(word_ & ~kRootOwner); }
  SelfWord enterForeign() const { return SelfWord(word_ + kForeignOne); }
  SelfWord leaveForeign() const { return SelfWord(word_ - kForeignOne); }

private:
  static constexpr int32_t kSlotBits = 0xff;
  static constexpr int32_t kRootOwner = 0x100;
  static constexpr int32_t kPendingShift = 9;
  static constexpr int32_t kPendingBits = 0x7f << kPendingShift;
  static constexpr int32_t kForeignOne = 1 << 16;
  static constexpr int32_t kForeignBits = 0x7fff * kForeignOne;
  // Set iff the thread is a worker, or inside a region.
  static constexpr int32_t kInPompBits = kSlotBits | kRootOwner | kForeignBits;

  int32_t word_;
};

#ifdef POMP_USE_PTHREAD_KEY
// Created at init. Until then no thread has a slot, and the key must not be
// read: its value would name a key of another library.
POMP_CONSTINIT pthread_key_t selfKey{};
POMP_CONSTINIT Atomic<int> selfKeyReady;

inline SelfWord SelfWord::current() {
  if (!selfKeyReady.load(Order::Acquire))
    return SelfWord();
  return SelfWord(int32_t(intptr_t(pthread_getspecific(selfKey))));
}
inline void SelfWord::set(SelfWord w) {
  checkPthread(pthread_setspecific(selfKey, (void *)intptr_t(w.word_)),
      "pthread_setspecific");
}
void SelfWord::createStorage() {
  checkPthread(pthread_key_create(&selfKey, nullptr), "pthread_key_create");
  selfKeyReady.store(1, Order::Release);
}
#else
POMP_CONSTINIT thread_local int32_t selfTls = 0;

inline SelfWord SelfWord::current() { return SelfWord(selfTls); }
inline void SelfWord::set(SelfWord w) { selfTls = w.word_; }
void SelfWord::createStorage() {}
#endif

// -----------------------------------------------------------------------------
// Invoking an outlined function
//
// Outlined functions are not variadic. Call each through its exact type, as
// libomp's portable kmp_invoke_microtask.cpp does, so that the call is valid
// on ABIs where variadic and fixed calls differ. A table, built at compile
// time, holds one caller per argument count.
// -----------------------------------------------------------------------------

template <unsigned... I>
struct IndexList {};
template <unsigned N, unsigned... I>
struct MakeIndexList : MakeIndexList<N - 1, N - 1, I...> {};
template <unsigned... I>
struct MakeIndexList<0, I...> {
  using Type = IndexList<I...>;
};

template <unsigned>
using ArgType = void *;

// Call `fn` with argv[I]..., through its exact type.
template <unsigned... I>
void invokeExact(Microtask fn, int32_t *gtid, int32_t *tid, void **argv,
    IndexList<I...>) {
  using Exact = void (*)(int32_t *, int32_t *, ArgType<I>...);
  reinterpret_cast<Exact>(fn)(gtid, tid, argv[I]...);
}

template <unsigned N>
void invokeWith(Microtask fn, int32_t *gtid, int32_t *tid, void **argv) {
  invokeExact(fn, gtid, tid, argv, typename MakeIndexList<N>::Type());
}

using Invoker = void (*)(Microtask, int32_t *, int32_t *, void **);

template <class List>
struct InvokerTable;
template <unsigned... N>
struct InvokerTable<IndexList<N...>> {
  static constexpr Invoker entries[] = {&invokeWith<N>...};
};

void invoke(Microtask fn, int32_t gtid, int32_t tid, int argc, void **argv) {
  using Table = InvokerTable<MakeIndexList<kMaxArgs + 1>::Type>;
  if (argc < 0 || argc > kMaxArgs)
    fatal("too many arguments to an outlined function");
  Table::entries[argc](fn, &gtid, &tid, argv);
}

// -----------------------------------------------------------------------------
// Data structures, all statically allocated
// -----------------------------------------------------------------------------

class Team;
class Thread;

// One entry of a thread's region stack.
struct Level {
  Team *team = nullptr; // nullptr for a serialized (team of one) region.
  int32_t tid = 0;
  int32_t nth = 0;
  uint64_t bar = 0; // Barriers this thread passed in this team, all regions.
};

// Synchronization words of a thread as a member of a team, on the thread's
// wake-up line (see Thread). `done` is written by the member and polled by the
// leader at the join;
// `arrive` is written by the member and polled by its parent in the barrier
// tree, and `release` the other way, so a line is shared by two threads at a
// time. Only members other than the leader use their line: the leader of a
// team stores no `done`, has no parent in the tree, and only reads and writes
// the lines of other members. A thread is such a member of at most one team at
// a time, as it leads any team it forks, and a level-1 member cannot be
// recruited at level 2. So each thread has one line, which it takes from team
// to team. The values stored are tagged with the team (see Team::tag), and
// waits test for equality, so that a value left by a previous team never
// matches; nothing needs to be reset.
struct MemberSync {
  Atomic<uint64_t> done;
  Atomic<uint64_t> arrive;
  Atomic<uint64_t> release;
};

// A team descriptor: written by its leader before waking the members, read
// by the members, and owned by the leader again after the join.
class Team {
public:
  // Leader: describe the work for the workers in `mask`, assign them, and wake
  // them. Members start at the leader's depth, at active level `level`.
  void launch(int32_t leaderSlot, uint64_t mask, int32_t level, int32_t depth,
      Microtask fn, int32_t argc, void **argv);
  // Leader: wait for every member to finish, then record the team's barrier
  // count, `bar` as the leader saw it, for the next region.
  void join(Thread &leader, uint64_t bar);
  // Member `tid`, a worker of slot `slot`: run the work, then report done.
  void runMember(Thread &self, int32_t slot, int32_t tid);
  // A barrier of the team, for the member at `top`.
  void barrier(Thread &self, Level &top);

  int32_t size() const { return nth_; }
  uint64_t barrierCount() const { return barSeq_; }
  // For a level-1 team: the workers that member `tid` prefers at level 2.
  uint64_t preferredFor(int32_t tid) const { return subMask_[tid]; }
  // Reuse cache of the recruitment: the workers of the last request of
  // `want` workers, if all of them are among `idle`, or else 0.
  uint64_t cachedPick(int want, uint64_t idle) const {
    return cachedReq_ == want && cachedMask_ && (cachedMask_ & ~idle) == 0
               ? cachedMask_
               : 0;
  }
  void remember(int want, uint64_t pick) {
    cachedReq_ = want;
    cachedMask_ = pick;
  }

private:
  template <class Cond>
  void wait(Thread &self, int32_t tid, Cond cond);
  uint64_t claimSleepers(uint64_t mask);
  void wakeMembers(uint64_t mask);
  void wakeLeader();
  int32_t leaderSlot() const;
  void partition();
  void treeBarrier(Thread &self, int32_t tid, int32_t nth, uint64_t k);
  // The sync line of member `tid` in this team.
  MemberSync &syncOf(int32_t tid) const;
  // Value `n` of one of this team's counters, as stored in sync lines: n
  // teams apart, plus the team's index, so that the values of two teams
  // never meet. Counters start at 0 and are bumped before use, so a value is
  // never 0, which a line holds at first.
  uint64_t tag(uint64_t n) const { return n * kTeams + uint64_t(index()); }
  // Index of this team, 0..kTeams-1: the leader's slot for a level-2 team,
  // kMaxThreads for the root's.
  int32_t index() const;

  // Work, written before the wake-up of the members.
  Microtask fn_ = nullptr;
  int32_t argc_ = 0;
  void *argv_[kMaxArgs] = {};
  int32_t nth_ = 0;      // Team size, leader included.
  int32_t level_ = 0;    // Active level of this team: 1 or 2.
  int32_t depth_ = 0;    // Leader's nesting depth before this region.
  uint64_t region_ = 0;  // Number of this region, for the members' done flags.
  uint64_t barSeq_ = 0;  // Barrier count before this region, for the tree.
  uint8_t slotOfTid_[kMaxThreads] = {};
  // Reuse cache: the last request and the workers it got.
  int32_t cachedReq_ = 0;
  uint64_t cachedMask_ = 0;
  // For a level-1 team: preferred workers for the level-2 team of each tid.
  uint64_t subMask_[kMaxThreads] = {};
  // Central barrier, for small teams: arrival count and generation.
  alignas(kCacheLine) Atomic<int> barCount_;
  alignas(kCacheLine) Atomic<uint64_t> barEpoch_;
  // Bit t is set while member t sleeps in a join or barrier wait, or is about
  // to. Written only by sleepers and their wakers, so it stays shared.
  alignas(kCacheLine) Atomic<uint64_t> sleepers_;
};

// A thread descriptor, one per slot.
class Thread {
public:
  // Values of the presence of a thread in the pool as this slot.
  static constexpr int kFree = 0;    // No thread in the pool as this slot.
  static constexpr int kInPool = 1;  // A thread is in the pool as this slot.
  static constexpr int kLeaving = 2; // The thread is returning to the host.

  // The calling thread's descriptor, or nullptr for a non-worker that does not
  // own slot 0, which is outside of any region. Entry points that receive a
  // `gtid` use this instead: a `gtid` is only what the caller says, and the
  // thread's own word cannot name another thread, nor a slot out of range.
  static Thread *current();
  // The calling thread's innermost region, or nullptr outside of any.
  static Level *currentLevel();

  void init() { post_.init(); }

  // Leader `gtid`: run `fn` as a region of `want` threads (0: the default).
  void fork(int32_t gtid, int32_t want, Microtask fn, int32_t argc,
      void **argv);

  // Recruiter: the assignment for the next wake-up, then the wake-up.
  void assign(Team *team, int32_t tid) {
    wake_.team = team;
    wake_.tid = tid;
    wake_.go.fetchAdd(1);
  }
  // Worker: the wake-up epoch, and the assignment it brings.
  uint64_t epoch(Order o = Order::SeqCst) const { return wake_.go.load(o); }
  Team &assignedTeam() const { return *wake_.team; }
  int32_t assignedTid() const { return wake_.tid; }

  // Region stack.
  void push(Team *team, int32_t tid, int32_t nth);
  void pop();
  Level *top();
  int32_t depth() const { return depth_; }
  int32_t active() const { return active_; }
  void setNesting(int32_t depth, int32_t active) {
    depth_ = depth;
    active_ = active;
  }

  // Sleep.
  void post() { post_.post(); }
  void waitPost() { post_.wait(); }
  void clearPost() { post_.clear(); }

  // Presence in the pool. `observed` receives the presence on failure.
  bool claimSlot(int &observed) {
    observed = kFree;
    return present_.compareExchangeWeak(observed, kInPool);
  }
  int presence(Order o) const { return present_.load(o); }
  void setPresence(int p, Order o = Order::SeqCst) { present_.store(p, o); }

  // Sync line of this thread as a member, other than the leader, of a team.
  MemberSync &sync() { return wake_.sync; }

private:
  // Wake-up and team words, on one line. They are used in turn: the recruiter
  // writes the assignment and then `go` only while the thread is idle, and
  // the thread's teams use `sync` only while it runs their work, from its
  // wake-up to the leader's read of its `done`, before which the leader does
  // not return it to the pool. So the recruiter and the team never meet on
  // the line, and the thread finds its assignment on the line of its wake-up.
  struct alignas(kCacheLine) WakeLine {
    // The epoch the worker waits on.
    Atomic<uint64_t> go;
    // Assignment for the next wake-up, written by the recruiter before `go`.
    Team *team = nullptr;
    int32_t tid = 0;
    // Join and tree barrier.
    MemberSync sync;
  };
  static_assert(sizeof(WakeLine) == kCacheLine, "one line");
  WakeLine wake_;

  // Owned by the thread itself, on lines of their own: the region stack is
  // written at every barrier, while the leader may poll `sync`.
  // Nesting level; past kMaxDepth, only counted.
  alignas(kCacheLine) int32_t depth_ = 0;
  int32_t active_ = 0; // Number of active levels, enclosing teams included.
  Level stack_[kMaxDepth];
  // kFree, kInPool or kLeaving. Read by another thread only when it enters
  // the pool as this slot.
  Atomic<int> present_;
  // Sleep: only after a spin, on lines of its own.
  alignas(kCacheLine) Post post_;
};

// The pool: which workers are in it and idle, sleeping or released, and the
// gate that lets them in.
class Pool {
public:
  int enter(int slot);
  int close();
  int open(int threadLimit);
  int waitEmpty();
  bool isClosed(Order o = Order::SeqCst) const { return closed_.load(o) != 0; }
  void closeGate() { closed_.store(1); }
  int size() const { return numWorkers_.load(); }

  // Recruit up to `want` idle workers for `team`, preferring `prefer`, and
  // return their mask; and return them after the join.
  uint64_t recruit(Team &team, int want, uint64_t prefer);
  void returnWorkers(uint64_t mask);
  // The idle workers, as of a relaxed load.
  uint64_t idle() const;
  // Post to the sleepers among `mask`; see "Wake-up and sleep".
  void wakeSleeping(uint64_t mask) { wakeSleepers(sleeping_.load() & mask); }

private:
  bool waitWork(Thread &t, uint64_t b, uint64_t &last);
  int wakeSleepers(uint64_t mask);
  void wakeReleased(int n);
  void releaseIdle();

  // Bit i is clear iff slot i is an idle worker in the pool.
  alignas(kCacheLine) Atomic<uint64_t> busy_{UINT64_MAX};
  // Bit i is set while worker i sleeps, or is about to.
  alignas(kCacheLine) Atomic<uint64_t> sleeping_;
  // Bit i is set from the release that claimed worker i until it leaves.
  alignas(kCacheLine) Atomic<uint64_t> released_;
  // 1 from a close until the next open: no thread may enter, and forks
  // run on their leader alone.
  alignas(kCacheLine) Atomic<int> closed_;
  Atomic<int> numWorkers_;
};

// Settings, fixed at init, and the initialization of pomp itself.
class Settings {
public:
  // Initialize pomp once per process, on the first call of any entry point.
  void ensureInit() {
    if (initialized_.load(Order::Acquire) != 2)
      initOnce();
  }
  // Set the thread limit, once per process.
  int setLimit(long limit);

  int limit() const { return limit_.load(Order::Relaxed); }
  uint64_t limitMask() const { return limitMask_.load(Order::Relaxed); }
  // Default team size of a fork at `active` enclosing active levels.
  int numThreads(int active) const {
    return nthreads_[active < kMaxActive ? active : kMaxActive - 1].load(
        Order::Relaxed);
  }
  int maxActive() const { return maxActive_; }
  long spin() const { return spin_; }
  int treeMin() const { return treeMin_; }

private:
  void initOnce();
  void init();

  Atomic<int> initialized_; // 0: no, 1: in progress, 2: done.
  int envNthreads_[kMaxActive] = {}; // OMP_NUM_THREADS, 0: unset.
  int maxActive_ = 0;
  // Settings that depend on the thread limit, set once: by init from
  // OMP_THREAD_LIMIT, or else by the first pomp_open_pool that gives one.
  // The pool stays closed until then, and the open that sets them publishes
  // them with its store to `closed`.
  Mutex limitMutex_;
  Atomic<int> limit_; // Root included; 0 until set.
  Atomic<uint64_t> limitMask_;
  Atomic<int> nthreads_[kMaxActive]; // Default team sizes.
  Atomic<int> limitMismatchSeen_;
  long spin_ = 0;
  int treeMin_ = 0; // Team size from which the tree barrier is used.
};

// Ownership of slot 0; see "Root ownership".
class Root {
public:
  enum class Entry {
    Foreign, // Another thread owns slot 0: run as a foreign region.
    Own,     // The thread runs on its own slot, as usual.
    Taken,   // The thread just took ownership of slot 0.
  };
  Entry enter(SelfWord w);
  void leave();

private:
  bool tryEnter(SelfWord w);
  void warnSecondRoot();

  // 1 while a non-worker thread, the root owner, runs regions as slot 0.
  alignas(kCacheLine) Atomic<int> active_;
  Atomic<int> secondRootSeen_;
};

POMP_CONSTINIT Settings settings;
POMP_CONSTINIT Pool pool;
POMP_CONSTINIT Root root;
POMP_CONSTINIT Thread threads[kMaxThreads];
// Team led by the root at level 1, and team led by slot i at level 2.
POMP_CONSTINIT Team rootTeam;
POMP_CONSTINIT Team slotTeams[kMaxThreads];

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

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
  for (int i = 0; i < kMaxActive; ++i) {
    int n = envNthreads_[i];
    nthreads_[i].store(n && n < limit ? n : int(limit));
  }
  limitMask_.store(bitsBelow(int(limit)));
  limit_.store(int(limit));
  return POMP_OK;
}

void Settings::init() {
  const char *s, *end;
  long v;

  limitMutex_.init();
  for (Thread &t : threads)
    t.init();
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
          "pomp: warning: OMP_THREAD_LIMIT=\"%s\" is not in [1, %d] and "
          "is ignored; the pool stays closed until pomp_open_pool "
          "gives a limit\n",
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

// -----------------------------------------------------------------------------
// Wake-up and sleep
//
// An idle worker spins on its `go` epoch and on its bit in `released`, then
// sleeps. To sleep, it sets its bit in `sleeping`, re-reads both, and waits for
// a post. A waker first publishes (bumps `go`, or sets the `released` bit),
// then reads `sleeping`, and posts only to workers whose bit it clears itself.
// All these accesses are sequentially consistent, so either the worker sees
// the new value or the waker sees the bit: no wake-up is lost. Exactly one of
// them clears the bit, so every post is consumed in the sleep it was meant for.
// Spinning workers never touch a mutex, and a waker of spinning workers makes
// one load of `sleeping` and no system call.
//
// A post only means "re-check": a waker may clear the bit of a later sleep
// than the one it meant to end. The worker then re-checks, finds nothing new,
// and sleeps again.
// -----------------------------------------------------------------------------

// Post to the sleepers among `mask` whose sleeping bit this call clears.
// Return the number of posts.
int Pool::wakeSleepers(uint64_t mask) {
  if (!mask)
    return 0;
  uint64_t m = sleeping_.fetchAnd(~mask) & mask;
  for (uint64_t r = m; r; r &= r - 1)
    threads[lowestSlot(r)].post();
  return popcount(m);
}

// Wake up to n sleeping workers that a release claimed. Used by each leaving
// worker, so that sleepers are woken as a tree, and not by the leader.
void Pool::wakeReleased(int n) {
  while (n > 0) {
    uint64_t s = sleeping_.load();
    s &= released_.load();
    if (!s)
      return;
    n -= wakeSleepers(lowestBits(s, n));
  }
}

// Wait for work or for release. Return true for work, with `last` updated to
// the new epoch, and false when a release has claimed this worker.
bool Pool::waitWork(Thread &t, uint64_t b, uint64_t &last) {
  for (long i = 0; i < settings.spin(); ++i) {
    uint64_t g = t.epoch(Order::Acquire);
    if (g != last) {
      last = g;
      return true;
    }
    if (released_.load(Order::Relaxed) & b)
      return false;
    relax();
  }
  uint64_t g;
  for (;;) {
    sleeping_.fetchOr(b);
    g = t.epoch();
    bool rel = (released_.load() & b) != 0;
    if (g != last || rel) {
      // A waker that cleared the bit has posted, or will: consume it.
      if (!(sleeping_.fetchAnd(~b) & b))
        t.waitPost();
      break;
    }
    t.waitPost();
    g = t.epoch();
    rel = (released_.load() & b) != 0;
    if (g != last || rel)
      break;
  }
  if (g == last)
    return false;
  // Work and release never come together: both claim the busy bit.
  last = g;
  return true;
}

// -----------------------------------------------------------------------------
// Waits inside a team (join, barriers)
//
// The same protocol as the idle wait, with the team's `sleepers` word in place
// of `sleeping`, and the bit of the waiter's tid. A thread is in at most one
// wait at a time, so it sleeps on its own slot's post, and every post is
// consumed by the wait that set the bit. A waker publishes what it waits for,
// then calls wakeMembers; a seq_cst fence on each side orders the publish
// against the read of `sleepers`, and the bit against the read of the
// condition.
// -----------------------------------------------------------------------------

// Conditions of the waits, as function objects.

// The flag holds exactly `value`.
class FlagIs {
public:
  FlagIs(const Atomic<uint64_t> &flag, uint64_t value)
      : flag_(flag), value_(value) {}
  bool operator()() const { return flag_.load(Order::Acquire) == value_; }

private:
  const Atomic<uint64_t> &flag_;
  uint64_t value_;
};

// The flag no longer holds `value`.
class FlagLeaves {
public:
  FlagLeaves(const Atomic<uint64_t> &flag, uint64_t value)
      : flag_(flag), value_(value) {}
  bool operator()() const { return flag_.load(Order::Acquire) != value_; }

private:
  const Atomic<uint64_t> &flag_;
  uint64_t value_;
};

// The thread is no longer leaving the pool.
class NotLeaving {
public:
  explicit NotLeaving(const Thread &t) : thread_(t) {}
  bool operator()() const {
    return thread_.presence(Order::Acquire) != Thread::kLeaving;
  }

private:
  const Thread &thread_;
};

// Wait, as member `tid` running on descriptor `self`, until `cond` holds:
// spin, then yield, which lets an oversubscribed waitee run, then sleep after
// POMP_SPIN iterations in all.
template <class Cond>
void Team::wait(Thread &self, int32_t tid, Cond cond) {
  long i = 0;
  uint64_t b = bit(tid);
  while (!cond()) {
    if (i++ < settings.spin()) {
      if (i < kYieldSpin)
        relax();
      else
        sched_yield();
      continue;
    }
    sleepers_.fetchOr(b);
    fence(Order::SeqCst);
    if (cond()) {
      // A waker that cleared the bit has posted, or will: consume it.
      if (!(sleepers_.fetchAnd(~b) & b))
        self.waitPost();
      break;
    }
    self.waitPost();
  }
}

// Claim the members in `mask` that sleep in a team wait. Return the sleepers
// whose bit this call cleared, for the caller to post to.
inline uint64_t Team::claimSleepers(uint64_t mask) {
  fence(Order::SeqCst);
  uint64_t s = sleepers_.load(Order::Relaxed) & mask;
  return s ? sleepers_.fetchAnd(~s) & s : 0;
}

inline void Team::wakeMembers(uint64_t mask) {
  for (uint64_t s = claimSleepers(mask); s; s &= s - 1)
    threads[slotOfTid_[lowestSlot(s)]].post();
}

// A member's last access to its team, at the join: wake the leader if it
// sleeps. The leader is found from the team's address, as the team may
// already be reused.
inline void Team::wakeLeader() {
  if (claimSleepers(1))
    threads[leaderSlot()].post();
}

// The slot of the leader of this team.
inline int32_t Team::leaderSlot() const {
  return this == &rootTeam ? 0 : int32_t(this - slotTeams);
}

inline int32_t Team::index() const {
  return this == &rootTeam ? kMaxThreads : int32_t(this - slotTeams);
}

// The member's slot comes from the map, written before the wake-up.
inline MemberSync &Team::syncOf(int32_t tid) const {
  return threads[slotOfTid_[tid]].sync();
}

// Spin on a condition outside of any team (reclaim), yielding when long.
template <class Cond>
void spinUntil(Cond cond) {
  for (long n = 0; !cond();) {
    if (++n < kYieldSpin)
      relax();
    else
      sched_yield();
  }
}

// -----------------------------------------------------------------------------
// Region stack
// -----------------------------------------------------------------------------

// Levels past kMaxDepth are only counted: forks there are serialized, so each
// of them is this entry. Never written, as its team has one thread.
POMP_CONSTINIT Level serialLevel = {nullptr, 0, 1, 0};

inline void Thread::push(Team *team, int32_t tid, int32_t nth) {
  if (depth_ >= kMaxDepth) {
    if (nth > 1)
      fatal("internal: active region past the maximum depth");
    depth_++;
    return;
  }
  Level &l = stack_[depth_];
  l.team = team;
  l.tid = tid;
  l.nth = nth;
  l.bar = team ? team->barrierCount() : 0;
  depth_++;
  if (nth > 1)
    active_++;
}

inline void Thread::pop() {
  if (depth_ <= 0)
    fatal("__kmpc_end_serialized_parallel without a matching "
          "__kmpc_serialized_parallel");
  depth_--;
  if (depth_ < kMaxDepth && stack_[depth_].nth > 1)
    active_--;
}

inline Level *Thread::top() {
  if (depth_ > kMaxDepth)
    return &serialLevel;
  return depth_ > 0 ? &stack_[depth_ - 1] : nullptr;
}

inline Thread *Thread::current() {
  SelfWord w = SelfWord::current();
  return w.hasDescriptor() ? &threads[w.slot()] : nullptr;
}

inline Level *Thread::currentLevel() {
  Thread *self = current();
  return self ? self->top() : nullptr;
}

// -----------------------------------------------------------------------------
// Root ownership
//
// Every thread that is not a worker in the pool reads slot 0, so two host
// threads running models at once would share slot 0's region stack and teams.
// The first of them to start a region (fork or serialized) at depth 0 owns slot
// 0 until that region ends, and marks itself in its per-thread word. Another
// non-worker that starts a region meanwhile is detected, whatever the timing:
// its region runs inline as a team of one, without touching slot 0, and a
// warning is printed once. Entry points that take a `gtid` ignore it and use
// the calling thread's own word, so a non-owner sees no enclosing region,
// which is exactly what a team of one needs.
// -----------------------------------------------------------------------------

void Root::warnSecondRoot() {
  if (!secondRootSeen_.exchange(1))
    fprintf(stderr, "pomp: warning: two host threads ran OpenMP regions at "
                    "once; the later ones run on one thread (see POMP.md)\n");
}

// Take root ownership for the calling non-worker of word `w`. Return false if
// another thread owns it.
inline bool Root::tryEnter(SelfWord w) {
  int expected = 0;
  // Acquire: see slot 0 as the previous owner left it.
  if (!active_.compareExchange(expected, 1))
    return false;
  SelfWord::set(w.withRootOwner());
  return true;
}

inline void Root::leave() {
  SelfWord::set(SelfWord::current().withoutRootOwner());
  active_.store(0, Order::Release);
}

// For a region about to start on the calling thread, whose word is `w`
// (pending bits clear): say whether it runs on its slot as usual, after taking
// root ownership, or as a foreign region, as another thread owns slot 0.
inline Root::Entry Root::enter(SelfWord w) {
  if (w.slot() || w.ownsRoot())
    return Entry::Own;
  if (tryEnter(w))
    return Entry::Taken;
  warnSecondRoot();
  return Entry::Foreign;
}

// Run `fn` inline as a foreign region of one thread.
void runForeign(Microtask fn, int32_t argc, void **argv) {
  SelfWord::set(SelfWord::current().enterForeign());
  invoke(fn, 0, 0, argc, argv);
  SelfWord::set(SelfWord::current().leaveForeign());
}

// -----------------------------------------------------------------------------
// Fork / join
// -----------------------------------------------------------------------------

uint64_t Pool::idle() const {
  return ~busy_.load(Order::Relaxed) & settings.limitMask();
}

uint64_t Pool::recruit(Team &team, int want, uint64_t prefer) {
  uint64_t old = busy_.load(Order::Relaxed);
  for (;;) {
    uint64_t idle = ~old & settings.limitMask();
    // Same request, same workers all idle: O(1).
    uint64_t pick = team.cachedPick(want, idle);
    if (!pick) {
      // Take what is idle now, preferred workers first; fewer is fine.
      pick = lowestBits(idle & prefer, want);
      pick |= lowestBits(idle & ~pick, want - popcount(pick));
    }
    if (pick == 0)
      return 0;
    if (busy_.compareExchangeWeak(
            old, old | pick, Order::Acquire, Order::Relaxed)) {
      if (popcount(pick) == want)
        team.remember(want, pick);
      return pick;
    }
  }
}

// Return the workers of a team. A release that ran meanwhile, from another
// host thread, did not see them idle; they are released here instead. Both
// sides are seq_cst, so one of them sees the other.
void Pool::returnWorkers(uint64_t mask) {
  busy_.fetchAnd(~mask);
  if (isClosed())
    releaseIdle();
}

// For a level-1 team, split the remaining idle workers into one contiguous
// group per member, which that member then prefers for its level-2 team, so
// that concurrent inner forks do not compete and their caches stay valid.
void Team::partition() {
  uint64_t spare = pool.idle();
  int n = popcount(spare);
  for (int t = 0; t < nth_; ++t) {
    int k = n / nth_ + (t < n % nth_ ? 1 : 0);
    subMask_[t] = lowestBits(spare, k);
    spare &= ~subMask_[t];
  }
}

void Team::launch(int32_t leader, uint64_t mask, int32_t level, int32_t depth,
    Microtask fn, int32_t argc, void **argv) {
  fn_ = fn;
  argc_ = argc;
  memcpy(argv_, argv, sizeof(void *) * size_t(argc));
  nth_ = 1 + popcount(mask);
  level_ = level;
  depth_ = depth;
  region_++;
  slotOfTid_[0] = uint8_t(leader);
  barCount_.store(0, Order::Relaxed);
  if (level_ == 1 && settings.maxActive() > 1)
    partition();

  // Assign and wake the members; post only to those that sleep. The whole
  // map is written first: members use it to wake each other.
  int32_t tid = 1;
  for (uint64_t m = mask; m; ++tid, m &= m - 1)
    slotOfTid_[tid] = uint8_t(lowestSlot(m));
  for (tid = 1; tid < nth_; ++tid)
    threads[slotOfTid_[tid]].assign(this, tid);
  pool.wakeSleeping(mask);
}

// Poll each member's done flag. Plain loads of distinct lines overlap, where
// increments of one shared counter serialize. A long wait sleeps; each member
// wakes the leader after its done flag.
void Team::join(Thread &leader, uint64_t bar) {
  for (int32_t tid = 1; tid < nth_; ++tid)
    wait(leader, 0, FlagIs(syncOf(tid).done, tag(region_)));
  // Every member passed the same barriers: the leader's count is the team's
  // for the next region. Written after the join, as members read it on entry.
  barSeq_ = bar;
}

void Team::runMember(Thread &self, int32_t slot, int32_t tid) {
  uint64_t done = tag(region_);
  MemberSync &sync = self.sync();
  // Continue the leader's nesting: same depth, same active levels.
  self.setNesting(depth_, level_ - 1);
  self.push(this, tid, nth_);
  invoke(fn_, slot, tid, argc_, argv_);
  self.pop();
  // After this store, the leader may reuse the team; only the wake-up of
  // the leader follows, which allows it. The line keeps `done` until the
  // leader reads it: the thread joins no other team before the leader returns
  // it to the pool, after the join.
  sync.done.store(done, Order::Release);
  wakeLeader();
}

void Thread::fork(
    int32_t gtid, int32_t want, Microtask fn, int32_t argc, void **argv) {
  int32_t limit = settings.limit();
  if (want <= 0)
    want = settings.numThreads(active_);
  if (want > limit)
    want = limit;

  // Serialized: one thread asked for, no active level left, nested too
  // deeply, or pool closed.
  Team *team = nullptr;
  uint64_t mask = 0;
  if (want > 1 && active_ < settings.maxActive() && depth_ < kMaxDepth &&
      !pool.isClosed(Order::Acquire)) {
    uint64_t prefer;
    if (active_ == 0) {
      if (gtid != 0)
        fatal("a worker started a parallel region outside of a team");
      team = &rootTeam;
      prefer = settings.limitMask();
    } else {
      team = &slotTeams[gtid];
      Level *t = top();
      prefer = t->team ? t->team->preferredFor(t->tid) : 0;
    }
    mask = pool.recruit(*team, want - 1, prefer);
  }
  if (mask == 0) {
    push(nullptr, 0, 1);
    invoke(fn, gtid, 0, argc, argv);
    pop();
    return;
  }

  // Members start at the leader's depth and level. Run as tid 0, then join.
  team->launch(gtid, mask, active_ + 1, depth_, fn, argc, argv);
  push(team, 0, team->size());
  invoke(fn, gtid, 0, argc, argv);
  uint64_t bar = top()->bar;
  pop();
  team->join(*this, bar);
  pool.returnWorkers(mask);
}

// The fork of the calling thread, after its arguments are collected.
void forkCall(Microtask fn, int32_t argc, void **argv) {
  SelfWord w = SelfWord::current();
  int32_t want = w.pending();
  if (want)
    SelfWord::set(w = w.withoutPending());
  if (w.inForeign()) {
    runForeign(fn, argc, argv);
    return;
  }
  Root::Entry entry = root.enter(w);
  if (entry == Root::Entry::Foreign) {
    runForeign(fn, argc, argv);
    return;
  }
  threads[w.slot()].fork(w.slot(), want, fn, argc, argv);
  if (entry == Root::Entry::Taken)
    root.leave();
}

// -----------------------------------------------------------------------------
// The pool
// -----------------------------------------------------------------------------

int Pool::enter(int slot) {
  int limit = settings.limit();
  if (!limit)
    return POMP_ERR_CLOSED; // Closed until an open gives the limit.
  if (slot < 1 || slot >= limit)
    return POMP_ERR_SLOT;
  // A thread in the pool cannot enter it again, and the leader cannot enter
  // it from inside its regions.
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  Thread &self = threads[slot];
  uint64_t b = bit(slot);
  // Take the slot. If its previous thread was released but has not left yet,
  // wait for it: this costs this thread, not the leader. A slot whose thread
  // has not been released is in use.
  for (;;) {
    int observed;
    if (self.claimSlot(observed))
      break;
    if (observed == Thread::kInPool && !(released_.load() & b))
      return POMP_ERR_SLOT;
    sched_yield();
  }

  SelfWord::set(SelfWord(slot));
  self.setNesting(0, 0);
  self.clearPost();
  uint64_t epoch = self.epoch();

  // Become idle, hence recruitable, then check the gate. Release closes the
  // gate before it claims idle workers, so either this thread sees it closed
  // or the release claims it.
  busy_.fetchAnd(~b);
  if (isClosed()) {
    if (!(busy_.fetchOr(b) & b)) {
      // Nobody claimed this slot: back out.
      SelfWord::set(SelfWord());
      self.setPresence(Thread::kFree, Order::Release);
      return POMP_ERR_CLOSED;
    }
    // Claimed between the two steps, by a release or by a fork after an
    // open: serve as any worker.
  }
  numWorkers_.fetchAdd(1);

  while (waitWork(self, b, epoch))
    self.assignedTeam().runMember(self, slot, self.assignedTid());

  // Released. The busy bit stays set until a thread enters as this slot.
  self.setNesting(0, 0);
  self.setPresence(Thread::kLeaving);
  released_.fetchAnd(~b);
  wakeReleased(2);
  SelfWord::set(SelfWord());
  numWorkers_.fetchSub(1);
  self.setPresence(Thread::kFree, Order::Release);
  return POMP_OK;
}

// Claim every idle worker for a release, as a fork would, and start their
// wake-up. Called by a release after it closed the gate, and by a leader that
// returns its team's workers while the gate is closed.
void Pool::releaseIdle() {
  uint64_t old = busy_.load(), idle;
  do {
    idle = ~old & settings.limitMask() & ~uint64_t(1);
  } while (idle && !busy_.compareExchangeWeak(old, old | idle));
  if (!idle)
    return;
  released_.fetchOr(idle);
  // Spinning workers see their bit and leave; each leaving worker wakes
  // sleeping ones. Only if every claimed worker sleeps is there nobody to start
  // that chain: then post to one.
  if ((sleeping_.load() & idle) == idle)
    wakeSleepers(idle & (~idle + 1));
}

int Pool::close() {
  // Not from a worker, nor from inside the regions of the calling thread.
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  // Close the gate first, so an entry from now on either backs out or is
  // claimed just below.
  closeGate();
  releaseIdle();
  return POMP_OK;
}

int Pool::waitEmpty() {
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  int limit = settings.limit();
  // Wake any claimed worker still asleep, until all have left.
  while (released_.load() & settings.limitMask()) {
    wakeReleased(kMaxThreads);
    sched_yield();
  }
  for (int i = 1; i < limit; ++i)
    spinUntil(NotLeaving(threads[i]));
  return POMP_OK;
}

int Pool::open(int threadLimit) {
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  // Whether the pool is open or closed, the outcome is the same: open with
  // the process's limit, or POMP_ERR_LIMIT with the pool unchanged.
  int rc = settings.setLimit(threadLimit);
  if (rc != POMP_OK)
    return rc;
  closed_.store(0);
  return POMP_OK;
}

// -----------------------------------------------------------------------------
// Barriers
// -----------------------------------------------------------------------------

// Tree barrier: tid t is the parent of tids ARITY*t+1 .. ARITY*t+ARITY. Each
// thread waits for its children to arrive, reports its own arrival to its
// parent and waits for the parent's release, then releases its children.
// Every flag is written by one thread and read by one other, so no line is
// contended; the depth is log_ARITY(nth).
//
// Waits test for equality, as lines move from team to team (see MemberSync).
// No flag can pass the value its reader waits for: a child publishes the next
// `arrive` only after this barrier's `release`, which its parent sets only
// after this barrier's `arrive`.
void Team::treeBarrier(Thread &self, int32_t tid, int32_t nth, uint64_t k) {
  int32_t first = kTreeArity * tid + 1;
  int32_t last = first + kTreeArity;
  uint64_t children = 0;
  uint64_t v = tag(k);
  if (last > nth)
    last = nth;
  for (int32_t c = first; c < last; ++c) {
    wait(self, tid, FlagIs(syncOf(c).arrive, v));
    children |= bit(c);
  }
  if (tid != 0) {
    MemberSync &sync = self.sync();
    sync.arrive.store(v, Order::Release);
    wakeMembers(bit((tid - 1) / kTreeArity));
    wait(self, tid, FlagIs(sync.release, v));
  }
  for (int32_t c = first; c < last; ++c)
    syncOf(c).release.store(v, Order::Release);
  if (children)
    wakeMembers(children);
}

void Team::barrier(Thread &self, Level &top) {
  top.bar++;
  if (top.nth >= settings.treeMin()) {
    treeBarrier(self, top.tid, top.nth, top.bar);
    return;
  }
  uint64_t epoch = barEpoch_.load(Order::Acquire);
  if (barCount_.fetchAdd(1, Order::AcqRel) == top.nth - 1) {
    barCount_.store(0, Order::Relaxed);
    barEpoch_.store(epoch + 1, Order::Release);
    wakeMembers(bitsBelow(top.nth) & ~bit(top.tid));
  } else {
    wait(self, top.tid, FlagLeaves(barEpoch_, epoch));
  }
}

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
  Level *top = Thread::currentLevel();
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
  UT nth = top ? UT(top->nth) : 1;
  UT tid = top ? UT(top->tid) : 0;
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

} // namespace

// -----------------------------------------------------------------------------
// Host interface: the pool (pomp.h)
// -----------------------------------------------------------------------------

int pomp_get_thread_limit(void) {
  settings.ensureInit();
  return settings.limit();
}

int pomp_open_pool(int thread_limit) {
  settings.ensureInit();
  return pool.open(thread_limit);
}

int pomp_enter_pool(int slot) {
  settings.ensureInit();
  return pool.enter(slot);
}

int pomp_close_pool(void) {
  settings.ensureInit();
  return pool.close();
}

int pomp_wait_pool_empty(void) {
  settings.ensureInit();
  return pool.waitEmpty();
}

int pomp_is_pool_open(void) { return !pool.isClosed(); }

int pomp_pool_size(void) { return pool.size(); }

// -----------------------------------------------------------------------------
// __kmpc entry points
// -----------------------------------------------------------------------------

extern "C" {

int32_t __kmpc_global_thread_num(void *) {
  return SelfWord::current().slot();
}

void __kmpc_fork_call(void *, int32_t argc, Microtask fn, ...) {
  void *argv[kMaxArgs];
  va_list ap;
  settings.ensureInit();
  if (argc < 0 || argc > kMaxArgs)
    fatal("too many arguments to an outlined function");
  va_start(ap, fn);
  for (int32_t i = 0; i < argc; ++i)
    argv[i] = va_arg(ap, void *);
  va_end(ap);
  forkCall(fn, argc, argv);
}

void __kmpc_push_num_threads(void *, int32_t, int32_t nth) {
  settings.ensureInit();
  if (nth < 0)
    nth = 0;
  else if (nth > kMaxThreads)
    nth = kMaxThreads;
  SelfWord::set(SelfWord::current().withPending(nth));
}

void __kmpc_push_proc_bind(void *, int32_t, int32_t) {}

void __kmpc_serialized_parallel(void *, int32_t) {
  settings.ensureInit();
  SelfWord w = SelfWord::current();
  if (w.pending())
    SelfWord::set(w = w.withoutPending());
  if (w.inForeign() || root.enter(w) == Root::Entry::Foreign) {
    SelfWord::set(SelfWord::current().enterForeign());
    return;
  }
  Thread::current()->push(nullptr, 0, 1);
}

void __kmpc_end_serialized_parallel(void *, int32_t) {
  SelfWord w = SelfWord::current();
  if (w.inForeign()) {
    SelfWord::set(w.leaveForeign());
    return;
  }
  Thread *self = Thread::current();
  if (!self)
    fatal("__kmpc_end_serialized_parallel without a matching "
          "__kmpc_serialized_parallel");
  self->pop();
  if (self == &threads[0] && self->depth() == 0)
    root.leave();
}

void __kmpc_barrier(void *, int32_t) {
  Thread *self = Thread::current();
  Level *top = self ? self->top() : nullptr;
  if (!top || top->nth == 1)
    return;
  top->team->barrier(*self, *top);
}

void __kmpc_for_static_init_4(void *, int32_t, int32_t sched, int32_t *plast,
    int32_t *plower, int32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk) {
  staticInit<int32_t, int32_t, uint32_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_4u(void *, int32_t, int32_t sched, int32_t *plast,
    uint32_t *plower, uint32_t *pupper, int32_t *pstride, int32_t incr,
    int32_t chunk) {
  staticInit<uint32_t, int32_t, uint32_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_8(void *, int32_t, int32_t sched, int32_t *plast,
    int64_t *plower, int64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk) {
  staticInit<int64_t, int64_t, uint64_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_8u(void *, int32_t, int32_t sched, int32_t *plast,
    uint64_t *plower, uint64_t *pupper, int64_t *pstride, int64_t incr,
    int64_t chunk) {
  staticInit<uint64_t, int64_t, uint64_t>(
      sched, plast, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_fini(void *, int32_t) {}

// -----------------------------------------------------------------------------
// omp_* queries
// -----------------------------------------------------------------------------

int omp_get_thread_num(void) {
  Level *top = Thread::currentLevel();
  return top && !SelfWord::current().inForeign() ? top->tid : 0;
}

int omp_get_num_threads(void) {
  Level *top = Thread::currentLevel();
  return top && !SelfWord::current().inForeign() ? top->nth : 1;
}

// A foreign region is nested only in other foreign regions, all inactive.
int omp_get_level(void) {
  SelfWord w = SelfWord::current();
  if (w.inForeign())
    return w.foreignDepth();
  Thread *self = Thread::current();
  return self ? self->depth() : 0;
}

int omp_get_active_level(void) {
  Thread *self = Thread::current();
  return self && !SelfWord::current().inForeign() ? self->active() : 0;
}

int omp_in_parallel(void) { return omp_get_active_level() > 0; }

int omp_get_max_threads(void) {
  int active = omp_get_active_level();
  settings.ensureInit();
  int n = settings.numThreads(active);
  return n > 0 ? n : 1;
}

int omp_get_thread_limit(void) {
  settings.ensureInit();
  int n = settings.limit();
  return n > 0 ? n : 1;
}

int omp_get_max_active_levels(void) {
  settings.ensureInit();
  return settings.maxActive();
}

} // extern "C"
