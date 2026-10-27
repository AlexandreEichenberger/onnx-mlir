/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pompTypes.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the declarations of the region stack, team, thread,
// pool, settings and root, and the globals.
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
  // Once, at allocation, after teams is set: seed the unique ids with
  // index().
  void init();

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

  // The leader's slot: 0 for the root's team.
  int32_t leaderSlot() const;
  // Every member, leader included.
  SlotMask members() const { return workers_ | SlotMask::of(leaderSlot()); }
  // Level 1: split the idle pool threads among the members, as their
  // preferred workers at level 2.
  void leaderPartition();
  // Index in `teams`: the leader's slot at level 2, the limit for the root.
  int32_t index() const;
  // Whether a team of `nth` threads uses the tree barrier. Static, so a
  // barrier reads only the member's own entry.
  static bool usesTreeBarriers(int32_t nth);
  // Barriers in a region: initBarrier on entry, then barrier, then
  // leaderEndBarrier after the join.

  // Member `tid`, on entry: its barrier state. Unsynchronized: reads only
  // fields the wake-up published.
  MemberBarrier initBarrier(int32_t tid) const;
  // After the join: keep the last barrier id for the next region, in
  // the team, as the root team's leader may change between regions.
  void leaderEndBarrier(UniqueId currUniqueBarrierId);

  // New cache line (a Team is line-aligned). The fields up to central_ take
  // one line on s390x, two where a line is 128 bytes.

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

  // The calling thread's descriptor, or nullptr for a non-pool thread that
  // does not own slot 0. Entry points use this, not their `gtid` argument,
  // which is only what the caller claims.
  static Thread *current();
  // The calling thread's region stack, or nullptr.
  static RegionStack *currentRegions();

  // Initialize the descriptor, once per slot; later calls do nothing.
  void init();

  // Leader `gtid`: run `fn` on `want` threads (0: the default).
  void fork(
      int32_t gtid, int32_t want, Microtask fn, int32_t argc, void **argv);

  // This thread's slot.
  int32_t slot() const;

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
  void post() { post_.post(); }
  void waitPost() { post_.wait(); }
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

  // Workers preferred for this thread's level-2 team, set by the root's
  // Team::leaderPartition.
  void setPreferred(SlotMask slots) { wake_.preferred = slots; }
  SlotMask preferred() const { return wake_.preferred; }

private:
  // Wake-up and team words, on one line, used in turn: the recruiter writes
  // it only while the thread is idle, the team only while it runs the work.
  struct alignas(kCacheLine) WakeLine {
    // The epoch the pool thread waits on.
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
  // kFree, kInPool or kLeaving; also serializes entries as this slot.
  Atomic<int> present_;
  bool initialized_ = false; // See init.
  // New line: used only after a spin.
  alignas(kCacheLine) Post post_;
};

// The pool: which pool threads are idle, sleeping or released, and whether it
// is open.
class Pool {
public:
  // The host calls of pomp.h, in the order of a cycle: pomp_open_pool,
  // pomp_enter_pool, pomp_close_pool and pomp_wait_pool_empty.
  int open();
  int enter(int32_t slot);
  int close();
  int waitEmpty();
  bool isOpen(Order o = Order::SeqCst) const { return state_.load(o) == kOpen; }
  int size() const { return numPoolThreads_.load(); }
  // Fork that finds the pool not open: warn once if it was never opened.
  void warnIfNeverOpened();

  // Recruit up to `want` idle pool threads as workers, preferring `prefer`;
  // return them after the join.
  SlotMask recruit(Team &team, int want, SlotMask prefer);
  // The idle pool threads (relaxed).
  SlotMask idle() const;
  // Post to the sleepers among `workers`; see "Wake-up and sleep".
  void wakeSleeping(SlotMask workers);
  // After the join: make a team's workers idle again.
  void returnWorkers(SlotMask workers);

private:
  // Idle pool thread: wait for work (true) or for release (false).
  bool waitWork(Thread &t, SlotMask self, uint64_t &last);
  // Leaving pool thread: wake up to `n` released sleepers.
  void wakeReleased(int n);
  // Claim and release every idle pool thread.
  void releaseIdle();
  // Post to the sleepers among `workers` whose bit this call clears.
  int wakeSleepers(SlotMask workers);

  // Slot i is out iff it is an idle pool thread.
  alignas(kCacheLine) AtomicMask busy_{SlotMask::below(64)};
  // Slot i is in while pool thread i sleeps, or is about to.
  alignas(kCacheLine) AtomicMask sleeping_;
  // Slot i is in from the release that claimed pool thread i until it leaves.
  alignas(kCacheLine) AtomicMask released_;
  // kNeverOpened (zero) at start, then kOpen and kClosed by open and close.
  static constexpr int kNeverOpened = 0;
  static constexpr int kOpen = 1;
  static constexpr int kClosed = 2;
  alignas(kCacheLine) Atomic<int> state_;
  Atomic<int> numPoolThreads_;
  Atomic<int> neverOpenedSeen_;
};

// Settings, fixed at init, and the init of pomp.
class Settings {
public:
  // Init pomp, once, on the first call of any entry point.
  void ensureInit() {
    if (initialized_.load(Order::Acquire) != 2)
      initOnce();
  }
  // Set the thread limit, once per process; `fromEnv` if by OMP_THREAD_LIMIT.
  int setLimit(long limit, bool fromEnv);

  int limit() const { return limit_.load(Order::Relaxed); }
  // The limit, acquired before reading the descriptors.
  int limitAcquire() const { return limit_.load(Order::Acquire); }
  // Slots 0..limit-1.
  SlotMask limitSlots() const { return limitSlots_.load(Order::Relaxed); }
  // Warn once that no limit is set.
  void warnNoLimit();
  // Default team size of a fork at `active` enclosing active levels.
  int numThreads(int active) const {
    return nthreads_[active < kMaxActive ? active : kMaxActive - 1].load(
        Order::Relaxed);
  }
  int maxActive() const { return maxActive_; }
  long spin() const { return spin_; }
  int treeMin() const { return treeMin_; }

private:
  // Run init() once; concurrent callers wait for it.
  void initOnce();
  // Read the environment, and set the limit from OMP_THREAD_LIMIT.
  void init();

  Atomic<int> initialized_;          // 0: no, 1: in progress, 2: done.
  int envNthreads_[kMaxActive] = {}; // OMP_NUM_THREADS, 0: unset.
  int maxActive_ = 0;
  // Set once with the thread limit (setLimit).
  Mutex limitMutex_;
  Atomic<int> limit_; // Root included; 0 until set.
  AtomicMask limitSlots_;
  Atomic<int> nthreads_[kMaxActive]; // Default team sizes.
  bool limitFromEnv_ = false;        // Under limitMutex_.
  Atomic<int> limitMismatchSeen_;
  Atomic<int> noLimitSeen_;
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
  // How a region about to start on the caller, of word `w`, runs.
  Entry enter(SelfWord w);
  // The root, at the end of its outermost region: give up slot 0.
  void leave();

private:
  // Take ownership of slot 0 for the caller; false if already owned.
  bool tryEnter(SelfWord w);
  // Warn once that two host threads ran regions at once.
  void warnSecondRoot();

  // 1 while the root runs regions as slot 0.
  alignas(kCacheLine) Atomic<int> active_;
  Atomic<int> secondRootSeen_;
};

POMP_CONSTINIT Settings settings;
POMP_CONSTINIT Pool pool;
POMP_CONSTINIT Root root;
// The descriptors, zero-allocated by Settings::setLimit. teams[i] is the
// level-2 team led by slot i, teams[limit] the root's level-1 team.
POMP_CONSTINIT Thread *threads = nullptr;
POMP_CONSTINIT Team *teams = nullptr;

inline Team &rootTeam() { return teams[settings.limit()]; }
inline Team &slotTeam(int32_t slot) { return teams[slot]; }

inline int32_t Team::leaderSlot() const {
  int32_t i = index();
  return i == settings.limit() ? 0 : i;
}

// From the descriptors' addresses: no init needed.
inline int32_t Team::index() const { return int32_t(this - teams); }
inline int32_t Thread::slot() const { return int32_t(this - threads); }
