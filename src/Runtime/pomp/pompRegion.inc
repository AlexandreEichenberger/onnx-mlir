/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pompRegion.inc ---------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: parallel regions. The region stack, the root blocks,
// and fork/join.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Region stack
// -----------------------------------------------------------------------------

inline void RegionStack::pushActive(const Level &l) {
  if (depth_ >= kMaxDepth)
    fatal("internal: active region past the maximum depth");
  if (l.nth < 2)
    fatal("internal: active region of one thread");
  levels_[depth_] = l;
  depth_++;
  active_++;
}

inline void RegionStack::pushSerialized() {
  if (depth_ >= kMaxNesting)
    fatal("regions nested too deeply");
  if (depth_ < kMaxDepth)
    levels_[depth_] = Level{nullptr, 0, 1, MemberBarrier()};
  depth_++;
}

inline void RegionStack::pop() {
  if (depth_ <= 0)
    fatal("__kmpc_end_serialized_parallel without a matching "
          "__kmpc_serialized_parallel");
  depth_--;
  // A serialized entry is recorded with nth 1.
  if (depth_ < kMaxDepth && levels_[depth_].nth > 1)
    active_--;
}

// Without a recorded innermost entry (none, or past kMaxDepth), the region is
// serialized.

inline int32_t RegionStack::innermostTid() const {
  const Level *l = recordedInnermost();
  return l ? l->tid : 0;
}

inline int32_t RegionStack::innermostNth() const {
  const Level *l = recordedInnermost();
  return l ? l->nth : 1;
}

inline bool RegionStack::innermostIsActive() const {
  const Level *l = recordedInnermost();
  return l && l->nth > 1;
}

inline Level *RegionStack::activeInnermost() {
  return innermostIsActive() ? &levels_[depth_ - 1] : nullptr;
}

// -----------------------------------------------------------------------------
// Team: Fork / join
// -----------------------------------------------------------------------------

Level Team::levelFor(int32_t tid) const {
  Level l;
  l.team = const_cast<Team *>(this);
  l.tid = tid;
  l.nth = nth_;
  l.bar = initBarrier(tid);
  return l;
}

// For a level-1 team: split the remaining idle pool threads into one contiguous
// group per member, preferred for its level-2 team, so concurrent inner forks
// do not compete and their caches stay warm.
void Team::leaderPartition() {
  SlotMask spare = pool.idle();
  int n = spare.count();
  int32_t slot = leaderSlot();
  SlotMask rest = workers_;
  for (int32_t t = 0; t < nth_; ++t) {
    int k = n / nth_ + (t < n % nth_ ? 1 : 0);
    SlotMask group = spare.lowest(k);
    member(slot).setPreferred(group);
    spare -= group;
    if (rest) {
      slot = rest.lowest();
      rest -= SlotMask::of(slot);
    }
  }
}

void Team::leaderLaunch(SlotMask workers, int32_t level, int32_t depth,
    Microtask fn, int32_t argc, void **argv) {
  fn_ = fn;
  argc_ = argc;
  memcpy(argv_, argv, sizeof(void *) * size_t(argc));
  nth_ = 1 + workers.count();
  level_ = level;
  depth_ = depth;
  join_.leaderStart();
  workers_ = workers;
  if (level_ == 1 && settings.maxActive() > 1)
    leaderPartition();

  // Assign and wake the workers, posting only to sleepers. workers_ is set
  // first: members use it to wake each other.
  int32_t tid = 1;
  for (int32_t slot : workers)
    threads[slot].assign(this, tid++);
  pool.wakeSleeping(workers);
}

void Team::runWorker(Thread &self, int32_t slot, int32_t tid) {
  UniqueId doneUniqueId = TeamJoin::workerDoneValue(*this);
  // Continue the leader's nesting: same depth, same active levels.
  RegionStack &regions = self.regions();
  regions.startAt(depth_, level_ - 1);
  regions.pushActive(levelFor(tid));
  invoke(fn_, slot, tid, argc_, argv_);
  regions.pop();
  TeamJoin::workerFinish(*this, self, doneUniqueId);
}

void Team::leaderJoin(Thread &leader, UniqueId uniqueBarrierId) {
  TeamJoin::leaderWait(*this, leader);
  leaderEndBarrier(uniqueBarrierId);
}

void Thread::fork(int32_t want, Microtask fn, int32_t argc, void **argv) {
  // Until a limit is known it is 0, and so is every team size: the fork
  // serializes. The pool's descriptors exist once it has been opened: the
  // acquire load of its state below orders their allocation before their use.
  int limit = settings.limit();
  if (want <= 0)
    want = settings.numThreads(regions_.active());
  if (want > limit)
    want = limit;

  // Recruit, unless serialized: one thread, no active level left, too deep,
  // or pool not open.
  Team *team = nullptr;
  SlotMask workers;
  bool parallel = want > 1 && regions_.active() < settings.maxActive() &&
                  regions_.canNest();
  if (parallel && !pool.isOpen(Order::Acquire)) {
    pool.warnIfNeverOpened();
  } else if (parallel) {
    SlotMask prefer;
    if (regions_.active() == 0) {
      if (slot_ != 0)
        fatal("a pool thread started a parallel region outside of a team");
      team = levelOne_;
      prefer = settings.limitSlots();
    } else {
      team = levelTwo_;
      prefer = regions_.innermostIsActive() ? preferred() : SlotMask();
    }
    workers = pool.recruit(*team, want - 1, prefer);
  }
  // When we have no workers (reasons listed above), run serialized.
  if (!workers) {
    regions_.pushSerialized();
    invoke(fn, slot_, 0, argc, argv);
    regions_.pop();
    return;
  }

  // Now launch parallel work, run as tid 0, then join.
  team->leaderLaunch(
      workers, regions_.active() + 1, regions_.depth(), fn, argc, argv);
  regions_.pushActive(team->levelFor(0));
  invoke(fn, slot_, 0, argc, argv);
  UniqueId uniqueBarrierId =
      regions_.activeInnermost()->bar.currUniqueBarrierId;
  regions_.pop();
  team->leaderJoin(*this, uniqueBarrierId);
  pool.returnWorkers(workers);
}

// The fork of the calling thread, after its arguments are collected.
void forkCall(Microtask fn, int32_t argc, void **argv) {
  Thread &self = Thread::currentOrNewRoot();
  self.fork(self.takePending(), fn, argc, argv);
}
