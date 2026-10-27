/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pompTypes.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: the declarations of
// the region stack, the team, thread, pool, settings and root descriptors,
// the globals, and their small accessors. They build on the synchronization
// types of pompSyncTypes.inc; the code of each is in the file named in its
// comment.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Data structures: static, except for the descriptors, allocated once with
// the thread limit
// -----------------------------------------------------------------------------

// One entry of a thread's region stack.
struct Level {
  Team *team = nullptr; // nullptr for a serialized (team of one) region.
  int32_t tid = 0;
  int32_t nth = 0;
  MemberBarrier bar;
};

// A thread's regions, innermost on top, with the number of them that are
// active (more than 1 thread). Callers ask either for the innermost region's
// tid and size, which are 0 and 1 outside of any region and in a serialized
// one, or for its team membership, which only an active region has; how the
// entries are kept is not theirs to see. Regions past kMaxDepth are only
// counted: forks there are serialized. Past kMaxNesting, a push aborts.
//
// A worker starts at its leader's depth and active level (startAt): the
// entries below that depth are the leader's, on the leader's own stack, and
// this stack never pops below it. So `active` is the inherited count plus the
// active entries above it, not something the entries alone tell.
class RegionStack {
public:
  void startAt(int32_t depth, int32_t active) {
    depth_ = depth;
    active_ = active;
  }
  // Enter a region of a team of more than one thread, as the member that `l`
  // describes.
  void pushActive(const Level &l);
  // Enter a region of one thread.
  void pushSerialized();
  void pop();

  // The innermost region's tid and team size.
  int32_t topTid() const;
  int32_t topNth() const;
  // The innermost region's team membership, or nullptr if that region is
  // serialized or there is none. Its `bar` is written by the barriers of its
  // team; nothing else of it changes after push.
  Level *activeTop();
  bool topIsActive() const;

  int32_t depth() const { return depth_; }
  int32_t active() const { return active_; }
  // Whether a region forked now may be active: a slot is left for it.
  bool canNest() const { return depth_ < kMaxDepth; }

private:
  // The innermost recorded entry, or nullptr if none or past kMaxDepth. A
  // serialized entry is recorded as a team of one, for pop().
  const Level *recordedTop() const {
    return depth_ > 0 && depth_ <= kMaxDepth ? &levels_[depth_ - 1] : nullptr;
  }

  int32_t depth_ = 0;  // Nesting level; past kMaxDepth, only counted.
  int32_t active_ = 0; // Active levels, enclosing teams included.
  Level levels_[kMaxDepth];
};

// A team descriptor: written by its leader before waking the members, read
// by the members, and owned by the leader again after the join.
class Team {
public:
  // At allocation: seed the team's unique ids, of its joins and tree
  // barriers, with `index`, its index in `teams` (see "Unique ids"), which
  // index() cannot give yet, as `teams` is set after.
  void seedUniqueIds(int32_t index);

  // Team actions:

  // Leader: describe the work for the workers in `workers`, assign them, and
  // wake them. Members start at the leader's depth, at active level `level`.
  void launch(SlotMask workers, int32_t level, int32_t depth, Microtask fn,
      int32_t argc, void **argv);
  // Members: run the work as thread `self` in slot `slot` with assigned `tid`,
  // then report done.
  void runMember(Thread &self, int32_t slot, int32_t tid);
  // Members: execute a barrier for the member at `top`.
  void barrier(Thread &self, Level &top);
  // Leader: wait for every member to finish, then pass `uniqueBarrierId`,
  // the id of the leader's last barrier, to the next region.
  void join(Thread &leader, uint64_t uniqueBarrierId);

  // Team info:
  // The number of threads in this team, leader included.
  int32_t size() const { return nth_; }
  // The region-stack entry of member `tid` in this region.
  Level levelFor(int32_t tid) const;
  // Reuse cache of the recruitment: the workers of the last request of
  // `want` workers, if all of them are among `idle`, or else none.
  SlotMask cachedPick(int want, SlotMask idle) const;
  void remember(int want, SlotMask pick);

private:
  // The components use the team's members, tags and waits.
  friend class TeamJoin;
  friend class TeamCentralBarrier;
  friend class TeamTreeBarrier;

  int32_t leaderSlot() const;
  // Every member, leader included.
  SlotMask members() const { return workers_ | SlotMask::of(leaderSlot()); }
  void partition();
  // Index of this team in `teams`, below kTeams: the leader's slot for a
  // level-2 team, the limit for the root's.
  int32_t index() const;
  // Whether a team of `nth` threads uses the tree barrier, rather than the
  // central one. Static, so that a barrier tests the size in the member's
  // region-stack entry, and reads nothing of the team.
  static bool usesTreeBarriers(int32_t nth);
  // Barriers of the team, in three phases per region, each for the
  // algorithm of the team's size only (see "Barriers" in pompSync.inc):
  //   1. initMemberBarrier  each member, leader included, on entry;
  //   2. barrier            each member, at each barrier of the region;
  //   3. endLeaderBarrier   the leader, after the join.
  // Phase 1 needs no synchronization among members, and phase 3 is ordered
  // after every member's last barrier by the join.

  // Phase 1 of the barriers. Member `tid`, leader included, on entry to the
  // region: its barrier state. Members call it unsynchronized: it reads only
  // team fields that the wake-up published, and writes only the member's own
  // state.
  MemberBarrier initMemberBarrier(int32_t tid) const;
  // Phase 3 of the barriers. Leader, after the join: `currUniqueBarrierId`
  // is the id of its last barrier, from which the next region continues.
  // Kept in the team, not in the leader, because the leader's region-stack
  // entry is gone after the region, and the root team's leader may change
  // between regions. No member is in a barrier then, and none reads the id
  // before the next region's wake-up.
  void endLeaderBarrier(uint64_t currUniqueBarrierId);

  // New cache line, forced: a Team starts one, as it is line-aligned (some
  // of its members are). The fields up to central_ then take as many lines
  // as they need: one where a line is 256 bytes (s390x), two where it is 128.

  // Work, written before the wake-up of the members.
  Microtask fn_ = nullptr;
  int32_t argc_ = 0;
  void *argv_[kMaxArgs] = {};
  int32_t nth_ = 0;   // Team size, leader included.
  int32_t level_ = 0; // Active level of this team: 1 or 2.
  int32_t depth_ = 0; // Leader's nesting depth before this region.
  TeamJoin join_;
  // Read by members on entry: with the fields they read then, not on lines
  // of its own as the central barrier is.
  TeamTreeBarrier tree_;
  // The members other than the leader: tids 1..nth-1, in increasing order of
  // slots.
  SlotMask workers_;
  // Reuse cache: the last request and the workers it got.
  int32_t cachedReq_ = 0;
  SlotMask cachedPick_;
  // New cache lines, forced: two, as `count_` and `epoch_` each start one
  // (alignas in TeamCentralBarrier).
  TeamCentralBarrier central_;
  // New cache line, forced: `sleepers_` starts one (alignas in TeamWait).
  TeamWait wait_;
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
  static RegionStack *currentRegions();

  // Initialize the sleep, once per slot: by the first thread to enter as this
  // slot, before it becomes recruitable, and at allocation for slot 0. Other
  // threads post only to slots of the pool or of their teams, so never to one
  // before this. Entries as a slot are serialized by `present_`.
  void initPost() {
    if (!postReady_) {
      post_.init();
      postReady_ = true;
    }
  }

  // Leader `gtid`: run `fn` as a region of `want` threads (0: the default).
  void fork(
      int32_t gtid, int32_t want, Microtask fn, int32_t argc, void **argv);

  // This thread's slot.
  int32_t slot() const;

  // Recruiter: the assignment for the next wake-up, then the wake-up.
  void assign(Team *team, int32_t tid);
  // Worker: the wake-up epoch, and the assignment it brings.
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

  // Presence in the pool. `observed` receives the presence on failure.
  bool claimSlot(int &observed) {
    observed = kFree;
    return present_.compareExchangeWeak(observed, kInPool);
  }
  int presence(Order o) const { return present_.load(o); }
  void setPresence(int p, Order o = Order::SeqCst) { present_.store(p, o); }

  // Sync line of this thread as a member, other than the leader, of a team.
  MemberSync &sync() { return wake_.sync; }

  // The workers this thread prefers for a level-2 team: set by the root for
  // each member of a level-1 team, before its wake-up, and read by the member
  // in that region.
  void setPreferred(SlotMask slots) { wake_.preferred = slots; }
  SlotMask preferred() const { return wake_.preferred; }

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
    SlotMask preferred;
    int32_t tid = 0;
    // Join and tree barrier.
    MemberSync sync;
  };
  static_assert(sizeof(WakeLine) == kCacheLine, "one line");
  // New cache line, forced: a Thread starts one, as WakeLine is line-aligned,
  // and fills exactly one.
  WakeLine wake_;

  // New cache line, forced. Owned by the thread itself, on lines of their
  // own: the region stack is written at every barrier, while the leader may
  // poll `sync`. present_ and postReady_ follow on its last line.
  alignas(kCacheLine) RegionStack regions_;
  // kFree, kInPool or kLeaving. Read by another thread only when it enters
  // the pool as this slot.
  Atomic<int> present_;
  bool postReady_ = false; // See initPost.
  // New cache line, forced. Sleep: only after a spin, on lines of its own.
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
  // return them; and return them to the pool after the join.
  SlotMask recruit(Team &team, int want, SlotMask prefer);
  void returnWorkers(SlotMask workers);
  // The idle workers, as of a relaxed load.
  SlotMask idle() const;
  // Post to the sleepers among `workers`; see "Wake-up and sleep".
  void wakeSleeping(SlotMask workers);

private:
  bool waitWork(Thread &t, SlotMask self, uint64_t &last);
  int wakeSleepers(SlotMask workers);
  void wakeReleased(int n);
  void releaseIdle();

  // Slot i is out iff it is an idle worker in the pool.
  alignas(kCacheLine) AtomicMask busy_{SlotMask::below(64)};
  // Slot i is in while worker i sleeps, or is about to.
  alignas(kCacheLine) AtomicMask sleeping_;
  // Slot i is in from the release that claimed worker i until it leaves.
  alignas(kCacheLine) AtomicMask released_;
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
  // Warn once per process that no limit is set, so regions run on one thread.
  void warnNoLimit();
  // The limit, ordered before the reads of the descriptors that follow.
  int limitAcquire() const { return limit_.load(Order::Acquire); }
  // Slots 0..limit-1.
  SlotMask limitSlots() const { return limitSlots_.load(Order::Relaxed); }
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

  Atomic<int> initialized_;          // 0: no, 1: in progress, 2: done.
  int envNthreads_[kMaxActive] = {}; // OMP_NUM_THREADS, 0: unset.
  int maxActive_ = 0;
  // Settings that depend on the thread limit, set once: by init from
  // OMP_THREAD_LIMIT, or else by the first pomp_open_pool that gives one.
  // The pool stays closed until then, and the open that sets them publishes
  // them with its store to `closed`.
  Mutex limitMutex_;
  Atomic<int> limit_; // Root included; 0 until set.
  AtomicMask limitSlots_;
  Atomic<int> nthreads_[kMaxActive]; // Default team sizes.
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
// The descriptors, one per slot up to the thread limit, allocated when the
// limit is set (see Settings::setLimit) and published by its store of the
// limit. Before that, regions run as foreign ones and touch none of them.
// Zero bytes are the initial state of both classes: calloc gives it.
//
// teams[i] is the team that slot i leads at level 2, and teams[limit] the
// team that the root leads at level 1.
POMP_CONSTINIT Thread *threads = nullptr;
POMP_CONSTINIT Team *teams = nullptr;

inline Team &rootTeam() { return teams[settings.limit()]; }
inline Team &slotTeam(int32_t slot) { return teams[slot]; }

// The slot of the leader of this team.
inline int32_t Team::leaderSlot() const {
  int32_t i = index();
  return i == settings.limit() ? 0 : i;
}

// Computations from the addresses of the descriptors, so no init needed.
inline int32_t Team::index() const { return int32_t(this - teams); }
inline int32_t Thread::slot() const { return int32_t(this - threads); }
