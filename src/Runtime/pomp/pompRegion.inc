/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pompRegion.inc ---------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: parallel regions. The region stack, the ownership of
// slot 0, and fork/join.
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

inline Thread *Thread::current() {
  SelfWord w = SelfWord::current();
  return w.hasDescriptor() ? &threads[w.slot()] : nullptr;
}

inline RegionStack *Thread::currentRegions() {
  Thread *self = current();
  return self ? &self->regions() : nullptr;
}

// -----------------------------------------------------------------------------
// Root ownership
//
// All non-pool threads read slot 0, so two host threads running models at once
// would share its region stack and teams. The first to start an outermost
// region becomes the root, owning slot 0 until that region ends; another
// non-pool thread that starts one meanwhile runs it as a foreign region, and
// pomp warns once.
// -----------------------------------------------------------------------------

// Take root ownership for the caller, of word `w`; false if already owned.
inline bool Root::tryEnter(SelfWord w) {
  int expected = 0;
  // Acquire: see slot 0 as the previous owner left it.
  if (!active_.compareExchange(expected, 1))
    return false;
  SelfWord::set(w.withRootOwner());
  return true;
}

void Root::warnSecondRoot() {
  if (!secondRootSeen_.exchange(1))
    fprintf(stderr, "pomp: warning: two host threads ran OpenMP regions at "
                    "once; the later ones run on one thread (see POMP.md)\n");
}

// How a region about to start on the caller, of word `w` (no pending size),
// runs: on its own slot, on slot 0 as the new root, or as a foreign region.
inline Root::Entry Root::enter(SelfWord w) {
  if (w.slot() || w.ownsRoot())
    return Entry::Own;
  // No descriptors before the limit is set; the acquire orders their
  // allocation before their use.
  if (!settings.limitAcquire())
    return Entry::Foreign;
  if (tryEnter(w))
    return Entry::Taken;
  warnSecondRoot();
  return Entry::Foreign;
}

inline void Root::leave() {
  SelfWord::set(SelfWord::current().withoutRootOwner());
  active_.store(0, Order::Release);
}

// Run `fn` inline as a foreign region of one thread.
void runForeign(Microtask fn, int32_t argc, void **argv) {
  SelfWord::set(SelfWord::current().enterForeign());
  invoke(fn, 0, 0, argc, argv);
  SelfWord::set(SelfWord::current().leaveForeign());
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
    threads[slot].setPreferred(group);
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

void Thread::fork(
    int32_t gtid, int32_t want, Microtask fn, int32_t argc, void **argv) {
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
      if (gtid != 0)
        fatal("a pool thread started a parallel region outside of a team");
      team = &rootTeam();
      prefer = settings.limitSlots();
    } else {
      team = &slotTeam(gtid);
      prefer = regions_.innermostIsActive() ? preferred() : SlotMask();
    }
    workers = pool.recruit(*team, want - 1, prefer);
  }
  // When we have no workers (reasons listed above), run serialized.
  if (!workers) {
    regions_.pushSerialized();
    invoke(fn, gtid, 0, argc, argv);
    regions_.pop();
    return;
  }

  // Now launch parallel work, run as tid 0, then join.
  team->leaderLaunch(
      workers, regions_.active() + 1, regions_.depth(), fn, argc, argv);
  regions_.pushActive(team->levelFor(0));
  invoke(fn, gtid, 0, argc, argv);
  UniqueId uniqueBarrierId =
      regions_.activeInnermost()->bar.currUniqueBarrierId;
  regions_.pop();
  team->leaderJoin(*this, uniqueBarrierId);
  pool.returnWorkers(workers);
}

// The fork of the calling thread, after its arguments are collected.
void forkCall(Microtask fn, int32_t argc, void **argv) {
  SelfWord w = SelfWord::current();
  int32_t want = w.pending();
  // Check for cases where we cannot run in parallel, in which case we run the
  // work as foreign.
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
  // We are now forking the work with w.slot() as the leader.
  threads[w.slot()].fork(w.slot(), want, fn, argc, argv);
  // If the leader took the root for itself, now that we have finished the fork,
  // release the root for other level-1 region to take it again.
  if (entry == Root::Entry::Taken)
    root.leave();
}
