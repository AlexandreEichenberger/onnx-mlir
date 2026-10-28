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
  // the team, as the root team's leader may change between regions.
  void leaderEndBarrier(UniqueId currUniqueBarrierId);

  // New cache line (a Team is line-aligned). The fields up to central_ take
  // one line on s390x, two where a line is 128 bytes.

  // Set at allocation, never changed.
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

  // This thread's slot: 0 for a root.
  int32_t slot() const { return slot_; }
  // At allocation: the slot of this descriptor.
  void setSlot(int32_t slot) { slot_ = slot; }

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

  // Workers preferred for this thread's level-2 team, set by the root's
  // Team::leaderPartition.
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
  int32_t slot_ = 0; // Set at allocation.
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
  // Whether the pool has ever been opened.
  bool wasOpened() const { return state_.load() != kNeverOpened; }
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

// Settings: defaults, then pomp_set_option calls, then the environment.
// Each is read from the environment at init (the first call of any entry
// point); a set environment variable takes precedence over calls, unless
// POMP_IGNORE_ENV=1. See "Environment" in POMP.md.
class Settings {
public:
  // Where a setting's value comes from, by increasing precedence.
  enum Source : int8_t { kDefault, kCall, kEnv };
  // pomp_set_option: set setting `name` (its environment variable's name)
  // from `value`. POMP_ERR_LIMIT: unknown name or invalid value;
  // POMP_ERR_STATE: the setting cannot change now.
  int setOption(const char *name, const char *value, Source from);
  // OMP_DISPLAY_ENV: print the settings on stderr, once, at the first open
  // of the pool (when only POMP_SPIN can still change).
  void displayOnce();
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
  int maxActive() const { return maxActive_.load(Order::Relaxed); }
  // POMP_SPIN: a fixed spin in us, or -1 for auto, which adapts up to
  // spinMaxUs().
  bool spinAdaptive() const { return spinUs_.load(Order::Relaxed) < 0; }
  long spinUs() const { return spinUs_.load(Order::Relaxed); }
  long spinMaxUs() const { return spinMaxUs_.load(Order::Relaxed); }
  int treeMin() const { return treeMin_.load(Order::Relaxed); }

private:
  // Run init() once; concurrent callers wait for it.
  void initOnce();
  // Read the environment, and set the limit from OMP_THREAD_LIMIT.
  void init();
  // Parsers: return false if `s` is invalid; otherwise set the setting,
  // unless `apply` is false (validation only). Under optionMutex_.
  bool setNumThreads(const char *s, bool apply);
  bool setMaxActive(const char *s, bool apply);
  bool setTreeMin(const char *s, bool apply);
  bool setSpin(const char *s, bool apply);
  // Settings with a source, indexed as kOptions in pompInit.inc.
  static constexpr int kNumOptions = 4;

  Atomic<int> initialized_;          // 0: no, 1: in progress, 2: done.
  // Serializes setOption (calls, and init's reads of the environment).
  Mutex optionMutex_;
  Source sources_[kNumOptions] = {};
  bool ignoreEnv_ = false;           // POMP_IGNORE_ENV=1.
  int displayEnv_ = 0;               // OMP_DISPLAY_ENV: 0, 1 (true), 2 (verbose).
  Atomic<int> displayed_;
  int envNthreads_[kMaxActive] = {}; // OMP_NUM_THREADS, 0: unset.
  // Whether OMP_NUM_THREADS listed more than one level, for the default of
  // OMP_MAX_ACTIVE_LEVELS.
  bool nthreadsNested_ = false;
  Atomic<int> maxActive_;
  // Set once with the thread limit (setLimit).
  Mutex limitMutex_;
  Atomic<int> limit_; // Root included; 0 until set.
  AtomicMask limitSlots_;
  Atomic<int> nthreads_[kMaxActive]; // Default team sizes.
  bool limitFromEnv_ = false;        // Under limitMutex_.
  Atomic<int> limitMismatchSeen_;
  Atomic<int> noLimitSeen_;
  // Read at every wait: atomics, as POMP_SPIN may change while regions run.
  Atomic<long> spinUs_;
  Atomic<long> spinMaxUs_;
  Atomic<int> treeMin_; // Team size from which the tree barrier is used.
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

inline Thread &Team::member(int32_t slot) const {
  return slot ? threads[slot] : *leader_;
}
