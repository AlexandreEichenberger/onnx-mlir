/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pompIdentity.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: slot sets, and the
// per-thread word that names the calling thread's slot.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Slot sets
// -----------------------------------------------------------------------------

// A set of slots, one bit per slot of a 64-bit word. Slots are pool-wide
// ids: 0 for the root and 1..limit-1 for the workers (`threads[slot]`). Tids,
// the ids of members in their team, are not sets: the leader is tid 0, and the
// other members follow in increasing order of slots. Iterating visits the
// slots in increasing order, one step per slot in the set.
class SlotMask {
public:
  constexpr SlotMask() = default;
  static constexpr SlotMask of(int id) { return SlotMask(uint64_t(1) << id); }
  // Ids 0..n-1.
  static constexpr SlotMask below(int n) {
    return SlotMask(n >= 64 ? UINT64_MAX : (uint64_t(1) << n) - 1);
  }
  static constexpr SlotMask fromBits(uint64_t bits) { return SlotMask(bits); }
  constexpr uint64_t bits() const { return bits_; }

  constexpr explicit operator bool() const { return bits_ != 0; }
  int count() const { return __builtin_popcountll(bits_); }
  // The lowest id of a nonempty set.
  int lowest() const { return __builtin_ctzll(bits_); }
  // The n lowest ids, or all of them if fewer.
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
  // Set difference: the ids of this set that are not in `o`.
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
    int operator*() const { return __builtin_ctzll(rest_); }
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

// An atomic SlotMask. Operations take and return masks, as Atomic's do
// values; fetchSet and fetchClear add and remove slots and return the previous
// set.
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
