/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- pompIdentity.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: the per-thread word
// that names the calling thread's slot, and slot sets.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Thread identity: one per-thread word, SelfWord, which every entry point
// reads first to know who is calling. The terms it uses:
//
// - Worker: a thread that entered the pool with pomp_enter_pool(slot). It is
//   given its slot, 1..limit-1, on entry, and reads 0 again once it leaves.
//
// - Non-worker (host thread): any other thread, such as the application's
//   main thread or a thread of its own that runs a model. Its slot is always
//   0, so all non-workers share slot 0's descriptor, `threads[0]`.
//
// - Root: the one non-worker that currently owns slot 0, and so may use
//   `threads[0]`: its region stack, and the teams it leads at level 1. There
//   is at most one root at a time. No thread is the root between regions.
//   Ownership is taken when a non-worker starts a region (fork or serialized)
//   at depth 0: the first to swap Root::active_ from 0 to 1 becomes the root,
//   in whatever order the threads arrive. It is given up when that outermost
//   region ends. Any non-worker can then be the root of its next region, so
//   the root may change from region to region, but two are never root at once.
//
// - Foreign region: a region that a non-worker runs inline as a team of one,
//   without touching any descriptor. Its nesting is counted in this word
//   instead of in a region stack, and inside it every query and barrier sees
//   a team of one. A thread starts a foreign region:
//     * when another thread is the root: a second host thread running a model
//       at the same time (POMP.md, S9). pomp warns once;
//     * when the thread limit is not set yet, as there are no descriptors yet;
//     * for any region nested in a foreign region, even if the root has left
//       or the limit has been set meanwhile.
//   Like a region stack, the depth is at most kMaxNesting; deeper aborts.
//   Each foreign region ends by decrementing the depth. At depth 0 the thread
//   is no longer foreign, and its next region tries again to take the root.
//   Workers and the root never start foreign regions: they run on their slot.
//
// - Pending team size: the size __kmpc_push_num_threads gives the thread's
//   next fork, 0 if none. The next fork or serialized region consumes it.
//
// The pending size and foreign depth belong to the thread, not to its slot:
// non-workers share slot 0, so they cannot be kept in `threads[0]`.
//
// Layout of the word (bits): 0-7 slot, 8 root owner, 9-15 pending team size,
// 16-30 foreign depth. Zero is a non-worker that is in no region.
// -----------------------------------------------------------------------------

class SelfWord {
public:
  constexpr explicit SelfWord(int32_t word = 0) : word_(word) {}

  // The calling thread's word, and its update.
  static SelfWord current();
  static void set(SelfWord w);
  // Create the per-thread storage, once, before any thread has a slot.
  static void createStorage();

  // The thread's slot: 1..limit-1 for a worker, 0 for any non-worker.
  int32_t slot() const { return word_ & kSlotBits; }
  // A non-worker that is the root, inside its regions.
  bool ownsRoot() const { return (word_ & kRootOwner) != 0; }
  // A worker, or the root: the thread may use `threads[slot()]`. Any other
  // thread is a non-worker outside regions or in foreign ones.
  bool hasDescriptor() const { return (word_ & (kSlotBits | kRootOwner)) != 0; }
  // A worker, the root, or a thread in a foreign region: the pool calls of
  // pomp.h are refused from such a thread.
  bool inPomp() const { return (word_ & kInPompBits) != 0; }
  // The pending team size, 0 if none.
  int32_t pending() const { return (word_ & kPendingBits) >> kPendingShift; }
  // Inside a foreign region, and how deeply nested in foreign regions.
  bool inForeign() const { return (word_ & kForeignBits) != 0; }
  int32_t foreignDepth() const { return (word_ & kForeignBits) / kForeignOne; }

  SelfWord withPending(int32_t nth) const {
    return SelfWord((word_ & ~kPendingBits) | (nth << kPendingShift));
  }
  SelfWord withoutPending() const { return SelfWord(word_ & ~kPendingBits); }
  // Set by Root::tryEnter when the thread becomes the root, cleared by
  // Root::leave.
  SelfWord withRootOwner() const { return SelfWord(word_ | kRootOwner); }
  SelfWord withoutRootOwner() const { return SelfWord(word_ & ~kRootOwner); }
  // Start and end a foreign region: one level deeper, one level out.
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
  // Set iff the thread is a worker, or inside a region.
  static constexpr int32_t kInPompBits = kSlotBits | kRootOwner | kForeignBits;
  static_assert(kMaxNesting <= kForeignBits / kForeignOne,
      "the foreign depth field must hold kMaxNesting");

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
  // The subset made of the n smallest ids of this set, or the whole set if it
  // has fewer than n. E.g. {2, 5, 9, 12}.lowest(2) is {2, 5}.
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
  // Remove the slots of `m`; return the whole set before.
  SlotMask fetchClear(SlotMask m, Order o = Order::SeqCst) {
    return SlotMask::fromBits(bits_.fetchAnd(~m.bits(), o));
  }
  // Clear the slots of `m`; return those of them that this call cleared,
  // that is, those that were set until now. Of threads that race to clear a
  // slot, exactly one gets it back.
  SlotMask clearAndReport(SlotMask m, Order o = Order::SeqCst) {
    return fetchClear(m, o) & m;
  }
  // If the set is `expected`, replace it with `desired` and return true;
  // else, or on a spurious failure, load the set into `expected`.
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
