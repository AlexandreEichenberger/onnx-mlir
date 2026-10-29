/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ pompDescriptors.inc -------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the descriptors, a thread's region stack, its Team and
// Thread, and the globals of the pool's descriptors; and their small methods,
// which touch only their own fields.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

// One entry of a thread's region stack.
struct Level {
  Team *team = nullptr; // nullptr for a serialized (team of one) region.
  int32_t tid = 0;
  int32_t nth = 0;
  MemberBarrier bar;
};

// A thread's regions, innermost on top, and how many are active (more than
// one thread). Regions past kMaxDepth are only counted, and serialized.
//
// A worker starts at its leader's depth and active level (startAt): entries
// below are on the leader's stack, so `active` includes inherited levels.
class RegionStack {
public:
  // Worker, on entry to a team: start at the leader's depth and level.
  void startAt(int32_t depth, int32_t active) {
    depth_ = depth;
    active_ = active;
  }
  // Enter an active region, as the member `l` describes.
  void pushActive(const Level &l);
  // Enter a region of one thread.
  void pushSerialized();
  // Leave the innermost region.
  void pop();

  // The innermost region's tid and size: 0 and 1 outside regions.
  int32_t innermostTid() const;
  int32_t innermostNth() const;
  // The innermost region's entry if it is active, else nullptr.
  Level *activeInnermost();
  // Whether the innermost region is active.
  bool innermostIsActive() const;

  int32_t depth() const { return depth_; }
  int32_t active() const { return active_; }
  // Whether a region forked now may be active.
  bool canNest() const { return depth_ < kMaxDepth; }

private:
  // The innermost entry, or nullptr if none or past kMaxDepth.
  const Level *recordedInnermost() const {
    return depth_ > 0 && depth_ <= kMaxDepth ? &levels_[depth_ - 1] : nullptr;
  }

  int32_t depth_ = 0;  // Nesting level; past kMaxDepth, only counted.
  int32_t active_ = 0; // Active levels, enclosing teams included.
  Level levels_[kMaxDepth];
};

// A team: written by its leader before the wake-up, read by the members.
class Team {
public:
  // Once, at allocation: team `index` (see "Unique ids"), led by `leader`.
  void init(int32_t index, Thread &leader);

  // Give the work to `workers` and wake them.
  void leaderLaunch(SlotMask workers, int32_t level, int32_t depth,
      Microtask fn, int32_t argc, void **argv);
  // Run the work as `tid`, then report done.
  void runWorker(Thread &self, int32_t slot, int32_t tid);
  // Any member: a barrier.
  void barrier(Thread &self, Level &innermost);
  // Wait for the workers; the next region's barriers continue from
  // `uniqueBarrierId`.
  void leaderJoin(Thread &leader, UniqueId uniqueBarrierId);

  // Team size, leader included.
  int32_t size() const { return nth_; }
  // The member of slot `slot`: in a team, slot 0 is its leader, a root.
  Thread &member(int32_t slot) const;
  // Member `tid`'s region-stack entry.
  Level levelFor(int32_t tid) const;
  // Recruitment cache: the last pick for `want` workers if all are idle.
  SlotMask cachedPick(int want, SlotMask idle) const;
  // Record `pick` as the workers of a request of `want`, for cachedPick.
  void remember(int want, SlotMask pick);

private:
  friend class TeamJoin;
  friend class TeamCentralBarrier;
  friend class TeamTreeBarrier;

  // The leader's slot: 0 for a root's team.
  int32_t leaderSlot() const { return leaderSlot_; }
  // Every member, leader included.
  SlotMask members() const { return workers_ | SlotMask::of(leaderSlot_); }
  // Level 1: split the idle pool threads among the members, as their
  // preferred workers at level 2.
  void leaderPartition();
  // Whether a team of `nth` threads uses the tree barrier. Static, so a
  // barrier reads only the member's own entry.
  static bool usesTreeBarriers(int32_t nth);
  // Barriers in a region: initBarrier on entry, then barrier, then
  // leaderEndBarrier after the join.

  // Member `tid`, on entry: its barrier state. Unsynchronized: reads only
  // fields the wake-up published.
  MemberBarrier initBarrier(int32_t tid) const;
  // After the join: keep the last barrier id for the next region, in
  // the team, which each member reads on entry.
  void leaderEndBarrier(UniqueId currUniqueBarrierId);

  // New cache line (a Team is line-aligned). The fields up to central_ take
  // one line on s390x, two where a line is 128 bytes.

  // Set at allocation, never changed: a worker's late wake-up reads them
  // after its region (see "Roots" in pompRoots.inc).
  Thread *leader_ = nullptr;
  int32_t leaderSlot_ = 0;
  // Work, written before the wake-up of the members.
  Microtask fn_ = nullptr;
  int32_t argc_ = 0;
  void *argv_[kMaxArgs] = {};
  int32_t nth_ = 0;   // Team size, leader included.
  int32_t level_ = 0; // Active level of this team: 1 or 2.
  int32_t depth_ = 0; // Leader's nesting depth before this region.
  TeamJoin join_;
  // Read on entry, so with the other entry fields.
  TeamTreeBarrier tree_;
  // Tids 1..nth-1, in increasing order of slots.
  SlotMask workers_;
  // Recruitment cache.
  int cachedReq_ = 0;
  SlotMask cachedPick_;
  // Two new lines, for `count_` and `epoch_`.
  TeamCentralBarrier central_;
  // New line, for `sleepers_`.
  TeamWait wait_;
};

// A thread descriptor, one per slot.
class Thread {
public:
  // Presence of a thread in the pool as this slot.
  static constexpr int kFree = 0;
  static constexpr int kInPool = 1;
  static constexpr int kLeaving = 2; // Returning to the host.

  // The calling thread's descriptor, or nullptr for a thread that has none
  // (see Self). Entry points use this, not their `gtid` argument, which is
  // only what the caller claims.
  static Thread *current();
  // The calling thread's descriptor; a new root's if it has none.
  static Thread &currentOrNewRoot();
  // The calling thread's region stack, or nullptr.
  static RegionStack *currentRegions();
  // Whether the caller is a pool thread or inside a region: pool calls are
  // refused.
  static bool callerInPomp();

  // Initialize the descriptor, once per slot; later calls do nothing.
  void init();

  // As leader: run `fn` on `want` threads (0: the default).
  void fork(int32_t want, Microtask fn, int32_t argc, void **argv);

  // This thread's slot: 0 for a root.
  int32_t slot() const { return slot_; }
  // At allocation: the slot, and the teams this descriptor leads, at level 1
  // (a root's only) and at level 2.
  void initIdentity(int32_t slot, Team *levelOne, Team *levelTwo) {
    slot_ = slot;
    levelOne_ = levelOne;
    levelTwo_ = levelTwo;
  }

  // The team size pushed for the next region (__kmpc_push_num_threads):
  // set it, or take it, leaving none. 0: none.
  void setPending(int32_t nth) { pending_ = nth; }
  int32_t takePending() {
    int32_t nth = pending_;
    pending_ = 0;
    return nth;
  }

  // Recruiter: assign the next work, and wake the thread.
  void assign(Team *team, int32_t tid);
  // Pool thread: the wake-up epoch, and its assignment.
  uint64_t epoch(Order o = Order::SeqCst) const { return wake_.go.load(o); }
  Team &assignedTeam() const { return *wake_.team; }
  int32_t assignedTid() const { return wake_.tid; }

  // Region stack.
  RegionStack &regions() { return regions_; }
  const RegionStack &regions() const { return regions_; }

  // Sleep.
  void post() {
    POMP_STAT(kStatPosts);
    POMP_STAT_TIME(kStatPostNs, post_.post());
  }
  void waitPost() { POMP_STAT_TIME(kStatSleepNs, post_.wait()); }
  void clearPost() { post_.clear(); }

  // Take this slot if no thread is in the pool as it; `observed` gets the
  // presence on failure.
  bool claimSlot(int &observed) {
    observed = kFree;
    return present_.compareExchangeWeak(observed, kInPool);
  }
  int presence(Order o) const { return present_.load(o); }
  void setPresence(int p, Order o = Order::SeqCst) { present_.store(p, o); }

  // Flags of this thread as a worker of a team.
  WorkerSync &sync() { return wake_.sync; }

  // Workers preferred for this thread's level-2 team, set by the
  // Team::leaderPartition of its level-1 team.
  void setPreferred(SlotMask slots) { wake_.preferred = slots; }
  SlotMask preferred() const { return wake_.preferred; }

private:
  // Wake-up and team words, on one line, used in turn: the recruiter writes
  // it only while the thread is idle, the team only while it runs the work.
  struct alignas(kCacheLine) WakeLine {
    // The epoch the pool thread waits on. Incremented by 1 by
    // the recruiter, then by 1 by the team after the join. The pool thread
    // spins on it, then sleeps if it does not change.
    Atomic<uint64_t> go;
    // Assignment, written before `go`.
    Team *team = nullptr;
    SlotMask preferred;
    int32_t tid = 0;
    // Join and tree barrier.
    WorkerSync sync;
  };
  static_assert(sizeof(WakeLine) == kCacheLine, "one line");
  WakeLine wake_;

  // New line: private, and written at every barrier while the leader may
  // poll `sync`.
  alignas(kCacheLine) RegionStack regions_;
  // Set at allocation (initIdentity).
  int32_t slot_ = 0;
  Team *levelOne_ = nullptr;
  Team *levelTwo_ = nullptr;
  // Used by this thread only.
  int32_t pending_ = 0;
  // kFree, kInPool or kLeaving; also serializes entries as this slot.
  Atomic<int> present_;
  bool initialized_ = false; // See init.
  // New line: used only after a spin.
  alignas(kCacheLine) Post post_;

public:
  // The spins of this thread's waits, under POMP_SPIN=auto: waiting for
  // work, and waiting in a team (join, barrier). Used by this thread only.
  AdaptiveSpin idleSpin_, teamSpin_;
};

// The pool's descriptors, zero-allocated by Settings::allocate: threads[i] is
// pool thread i and teams[i] the level-2 team it leads. Index 0 is unused:
// each root has its own (Root).
POMP_CONSTINIT Thread *threads = nullptr;
POMP_CONSTINIT Team *teams = nullptr;

inline Thread &Team::member(int32_t slot) const {
  return slot ? threads[slot] : *leader_;
}

// -----------------------------------------------------------------------------
// Small methods
// -----------------------------------------------------------------------------

// Called on each entry as this slot, before it is recruitable, and at the
// allocation of a Root. Entries as a slot are serialized by `present_`.
// The post is initialized once per descriptor, never again: a waker of a
// previous occupant (or owner, for a Root) may still be inside its
// post() ("Roots", rule 2, in pompRoots.inc).
inline void Thread::init() {
  if (!initialized_) {
    post_.init();
    initialized_ = true;
  }
}

// Join and tree ids share a seed: they go to different flags. Once per team,
// at allocation: never again, even when a Root is reused ("Roots", rule
// 3).
inline void Team::init(int32_t index, Thread &leader) {
  leader_ = &leader;
  leaderSlot_ = leader.slot();
  join_.seed(index);
  tree_.seed(index);
}

inline Thread *Thread::current() { return Self::get(); }

inline RegionStack *Thread::currentRegions() {
  Thread *self = current();
  return self ? &self->regions() : nullptr;
}

inline bool Thread::callerInPomp() {
  Thread *self = current();
  return self && (self->slot() != 0 || self->regions().depth() > 0);
}

// The bump of `go` publishes the assignment (seq_cst).
inline void Thread::assign(Team *team, int32_t tid) {
  wake_.team = team;
  wake_.tid = tid;
  wake_.go.fetchAdd(1);
}

inline SlotMask Team::cachedPick(int want, SlotMask idle) const {
  return cachedReq_ == want && cachedPick_ && !(cachedPick_ - idle)
             ? cachedPick_
             : SlotMask();
}

inline void Team::remember(int want, SlotMask pick) {
  cachedReq_ = want;
  cachedPick_ = pick;
}
