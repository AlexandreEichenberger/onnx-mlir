/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pompIdentity.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the per-thread word (SelfWord) and slot sets (SlotMask).
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Thread identity: SelfWord, the per-thread word every entry point reads
// first.
//
// Threads, by their place in the pool:
// - Pool thread: a thread lent to the pool (pomp_enter_pool), as slot
//   1..limit-1. Between teams it waits idle in the pool.
// - Non-pool thread: any other thread. All of them read slot 0, which one of
//   them at a time may own (see "Root ownership" in pompRegion.inc).
// - Foreign region: a region that a non-pool thread runs inline, as a team
//   of one, touching no descriptor, with its nesting counted in this word:
//   while another thread owns slot 0, before the thread limit is set, and
//   inside another foreign region.
//
// Threads, by their place in a team (an active region):
// - Leader: the thread that forks the region, tid 0. It sets up the team,
//   wakes the workers, runs its share, and joins.
// - Worker: a pool thread recruited into the team, tid 1..nth-1. It leaves
//   after its done report, without waiting.
// - Member: any of the team's nth threads, leader included.
// A pool thread is idle, or a worker of one team; as a worker it may also
// lead a level-2 team.
//
// Functions take a `leader` or `worker` prefix when only that role calls
// them; unprefixed team functions are for any member.
//
// The pending team size (__kmpc_push_num_threads, consumed by the next fork
// or serialized region) and the foreign depth are per thread, not per slot,
// as non-pool threads share slot 0.
//
// Layout (bits): 0-7 slot, 8 root owner, 9-15 pending size, 16-30 foreign
// depth. Zero is a non-pool thread in no region.
// -----------------------------------------------------------------------------

class SelfWord {
public:
  constexpr explicit SelfWord(int32_t word = 0) : word_(word) {}

  // The calling thread's word, and its update.
  static SelfWord current();
  static void set(SelfWord w);
  // Create the per-thread storage, once, before any thread has a slot.
  static void createStorage();

  // The thread's slot: 1..limit-1 for a pool thread, 0 for any other.
  int32_t slot() const { return word_ & kSlotBits; }
  // A non-pool thread that owns slot 0, inside its regions.
  bool ownsRoot() const { return (word_ & kRootOwner) != 0; }
  // A pool thread, or the owner of slot 0: may use `threads[slot()]`.
  bool hasDescriptor() const { return (word_ & (kSlotBits | kRootOwner)) != 0; }
  // A pool thread, or inside a region: pool calls are refused.
  bool inPomp() const { return (word_ & kInPompBits) != 0; }
  // The pending team size, 0 if none.
  int32_t pending() const { return (word_ & kPendingBits) >> kPendingShift; }
  // Inside a foreign region, and how deeply nested in foreign regions.
  bool inForeign() const { return (word_ & kForeignBits) != 0; }
  int32_t foreignDepth() const { return (word_ & kForeignBits) / kForeignOne; }

  // Set or clear the pending team size.
  SelfWord withPending(int32_t nth) const {
    return SelfWord((word_ & ~kPendingBits) | (nth << kPendingShift));
  }
  SelfWord withoutPending() const { return SelfWord(word_ & ~kPendingBits); }
  // Set and cleared by Root::tryEnter and Root::leave.
  SelfWord withRootOwner() const { return SelfWord(word_ | kRootOwner); }
  SelfWord withoutRootOwner() const { return SelfWord(word_ & ~kRootOwner); }
  // Enter or leave a foreign region: one level deeper, or one out.
  SelfWord enterForeign() const {
    if (foreignDepth() >= kMaxNesting)
      fatal("regions nested too deeply");
    return SelfWord(word_ + kForeignOne);
  }
  SelfWord leaveForeign() const { return SelfWord(word_ - kForeignOne); }

private:
  static constexpr int32_t kSlotBits = 0xff;
  static constexpr int32_t kRootOwner = 0x100;
  static constexpr int32_t kPendingShift = 9;
  static constexpr int32_t kPendingBits = 0x7f << kPendingShift;
  static constexpr int32_t kForeignOne = 1 << 16;
  static constexpr int32_t kForeignBits = 0x7fff * kForeignOne;
  static constexpr int32_t kInPompBits = kSlotBits | kRootOwner | kForeignBits;
  static_assert(kMaxNesting <= kForeignBits / kForeignOne,
      "the foreign depth field must hold kMaxNesting");

  int32_t word_;
};

#ifdef POMP_USE_PTHREAD_KEY
// Created at init, and not read before: its value could be another library's.
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
// Slot sets
// -----------------------------------------------------------------------------

// A set of slots, one bit each in a 64-bit word. Slots are pool-wide ids (0
// for the owner of slot 0, 1..limit-1 for pool threads); tids are ids within
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
