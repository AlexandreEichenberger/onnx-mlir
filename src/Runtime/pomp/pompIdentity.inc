/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pompIdentity.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the calling thread's descriptor (Self) and slot sets
// (SlotMask).
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Thread identity: Self, the per-thread pointer every entry point reads first.
//
// Threads, by their place in the pool:
// - Pool thread: a thread lent to the pool (pomp_enter_pool), as slot
//   1..limit-1. Between teams it waits idle in the pool.
// - Root: any other thread, once it starts a region: OpenMP's initial
//   thread. Each root has a Root, the data it needs: a descriptor of its
//   own, of slot 0, and the two teams it leads (see "Roots" in
//   pompRoots.inc). Before its first region, a thread has no descriptor.
//
// Threads, by their place in a team (an active region):
// - Leader: the thread that forks the region, tid 0: OpenMP's primary thread
//   ("leader" pairs with "worker" and "member"). It sets up the team, wakes
//   the workers, runs its share, and joins.
// - Worker: a pool thread recruited into the team, tid 1..nth-1. It leaves
//   after its done report, without waiting.
// - Member: any of the team's nth threads, leader included.
// A pool thread is idle, or a worker of one team; as a worker it may also
// lead a level-2 team.
//
// Functions take a `leader` or `worker` prefix when only that role calls
// them; unprefixed team functions are for any member.
// -----------------------------------------------------------------------------

class Thread;

class Self {
public:
  // The calling thread's descriptor: its pool slot's while in the pool, its
  // root's otherwise, or nullptr if it has none.
  static Thread *get();
  static void set(Thread *t);
  // Create the per-thread storage, once, before any thread has a descriptor.
  static void createStorage();
};

#ifdef POMP_USE_PTHREAD_KEY
// Created at init, and not read before: its value could be another library's.
POMP_CONSTINIT pthread_key_t selfKey{};
POMP_CONSTINIT Atomic<int> selfKeyReady;

inline Thread *Self::get() {
  if (!selfKeyReady.load(Order::Acquire))
    return nullptr;
  return static_cast<Thread *>(pthread_getspecific(selfKey));
}
inline void Self::set(Thread *t) {
  checkPthread(pthread_setspecific(selfKey, t), "pthread_setspecific");
}
void Self::createStorage() {
  checkPthread(pthread_key_create(&selfKey, nullptr), "pthread_key_create");
  selfKeyReady.store(1, Order::Release);
}
#else
POMP_CONSTINIT thread_local Thread *selfTls = nullptr;

inline Thread *Self::get() { return selfTls; }
inline void Self::set(Thread *t) { selfTls = t; }
void Self::createStorage() {}
#endif

// -----------------------------------------------------------------------------
// Slot sets
// -----------------------------------------------------------------------------

// A set of slots, one bit each in a 64-bit word. Slots are pool-wide ids (0
// for a team's root leader, 1..limit-1 for pool threads); tids are ids within
// a team (leader 0, then the workers by increasing slot). Iteration visits
// slots in increasing order.
class SlotMask {
public:
  constexpr SlotMask() = default;
  // The set {id}.
  static constexpr SlotMask of(int32_t id) {
    return SlotMask(uint64_t(1) << id);
  }
  // Ids 0..n-1.
  static constexpr SlotMask below(int n) {
    return SlotMask(n >= 64 ? UINT64_MAX : (uint64_t(1) << n) - 1);
  }
  static constexpr SlotMask fromBits(uint64_t bits) { return SlotMask(bits); }
  constexpr uint64_t bits() const { return bits_; }

  constexpr explicit operator bool() const { return bits_ != 0; }
  int count() const { return __builtin_popcountll(bits_); }
  // The lowest id of a nonempty set.
  int32_t lowest() const { return __builtin_ctzll(bits_); }
  // The n smallest ids, or all if fewer: {2, 5, 9, 12}.lowest(2) is {2, 5}.
  SlotMask lowest(int n) const {
    uint64_t x = bits_, r = 0;
    for (; n > 0 && x; --n) {
      uint64_t low = x & (~x + 1);
      r |= low;
      x &= ~low;
    }
    return SlotMask(r);
  }

  constexpr SlotMask operator|(SlotMask o) const {
    return SlotMask(bits_ | o.bits_);
  }
  constexpr SlotMask operator&(SlotMask o) const {
    return SlotMask(bits_ & o.bits_);
  }
  // Set difference.
  constexpr SlotMask operator-(SlotMask o) const {
    return SlotMask(bits_ & ~o.bits_);
  }
  SlotMask &operator|=(SlotMask o) { return *this = *this | o; }
  SlotMask &operator-=(SlotMask o) { return *this = *this - o; }
  constexpr bool operator==(SlotMask o) const { return bits_ == o.bits_; }
  constexpr bool operator!=(SlotMask o) const { return bits_ != o.bits_; }

  class Iterator {
  public:
    explicit Iterator(uint64_t rest) : rest_(rest) {}
    int32_t operator*() const { return __builtin_ctzll(rest_); }
    Iterator &operator++() {
      rest_ &= rest_ - 1;
      return *this;
    }
    bool operator!=(Iterator o) const { return rest_ != o.rest_; }

  private:
    uint64_t rest_;
  };
  Iterator begin() const { return Iterator(bits_); }
  Iterator end() const { return Iterator(0); }

private:
  constexpr explicit SlotMask(uint64_t bits) : bits_(bits) {}
  uint64_t bits_ = 0;
};

// An atomic SlotMask; fetchSet and fetchClear return the previous set.
class AtomicMask {
public:
  constexpr AtomicMask() = default;
  constexpr explicit AtomicMask(SlotMask m) : bits_(m.bits()) {}

  SlotMask load(Order o = Order::SeqCst) const {
    return SlotMask::fromBits(bits_.load(o));
  }
  void store(SlotMask m, Order o = Order::SeqCst) { bits_.store(m.bits(), o); }
  SlotMask fetchSet(SlotMask m, Order o = Order::SeqCst) {
    return SlotMask::fromBits(bits_.fetchOr(m.bits(), o));
  }
  SlotMask fetchClear(SlotMask m, Order o = Order::SeqCst) {
    return SlotMask::fromBits(bits_.fetchAnd(~m.bits(), o));
  }
  // Clear `m`; return the slots this call cleared. Of racing clearers of a
  // slot, exactly one gets it.
  SlotMask clearAndReport(SlotMask m, Order o = Order::SeqCst) {
    return fetchClear(m, o) & m;
  }
  // As Atomic's: on failure, `expected` gets the current set.
  bool compareExchangeWeak(SlotMask &expected, SlotMask desired,
      Order success = Order::SeqCst, Order failure = Order::SeqCst) {
    uint64_t e = expected.bits();
    bool ok = bits_.compareExchangeWeak(e, desired.bits(), success, failure);
    expected = SlotMask::fromBits(e);
    return ok;
  }

private:
  Atomic<uint64_t> bits_;
};
