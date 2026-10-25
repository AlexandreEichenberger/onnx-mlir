/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pompRegion.inc ---------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: parallel regions.
// The region stack, the ownership of slot 0 by a host thread, and fork/join:
// launch, run and join of a team.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Region stack
// -----------------------------------------------------------------------------

inline void RegionStack::pushActive(const Level &l) {
  if (depth_ >= kMaxDepth || l.nth < 2)
    fatal("internal: active region past the maximum depth or of one thread");
  levels_[depth_] = l;
  depth_++;
  active_++;
}

inline void RegionStack::pushSerialized() {
  if (depth_ < kMaxDepth)
    levels_[depth_] = Level{nullptr, 0, 1, MemberBarrier()};
  depth_++;
}

inline void RegionStack::pop() {
  if (depth_ <= 0)
    fatal("__kmpc_end_serialized_parallel without a matching "
          "__kmpc_serialized_parallel");
  depth_--;
  if (depth_ < kMaxDepth && levels_[depth_].nth > 1)
    active_--;
}

inline int32_t RegionStack::topTid() const {
  const Level *l = recordedTop();
  return l ? l->tid : 0;
}

inline int32_t RegionStack::topNth() const {
  const Level *l = recordedTop();
  return l ? l->nth : 1;
}

inline bool RegionStack::topIsActive() const {
  const Level *l = recordedTop();
  return l && l->nth > 1;
}

inline Level *RegionStack::activeTop() {
  return topIsActive() ? &levels_[depth_ - 1] : nullptr;
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
// Every thread that is not a worker in the pool reads slot 0, so two host
// threads running models at once would share slot 0's region stack and teams.
// The first of them to start a region (fork or serialized) at depth 0 owns slot
// 0 until that region ends, and marks itself in its per-thread word. Another
// non-worker that starts a region meanwhile is detected, whatever the timing:
// its region runs inline as a team of one, without touching slot 0, and a
// warning is printed once. Entry points that take a `gtid` ignore it and use
// the calling thread's own word, so a non-owner sees no enclosing region,
// which is exactly what a team of one needs.
// -----------------------------------------------------------------------------

void Root::warnSecondRoot() {
  if (!secondRootSeen_.exchange(1))
    fprintf(stderr, "pomp: warning: two host threads ran OpenMP regions at "
                    "once; the later ones run on one thread (see POMP.md)\n");
}

// Take root ownership for the calling non-worker of word `w`. Return false if
// another thread owns it.
inline bool Root::tryEnter(SelfWord w) {
  int expected = 0;
  // Acquire: see slot 0 as the previous owner left it.
  if (!active_.compareExchange(expected, 1))
    return false;
  SelfWord::set(w.withRootOwner());
  return true;
}

inline void Root::leave() {
  SelfWord::set(SelfWord::current().withoutRootOwner());
  active_.store(0, Order::Release);
}

// For a region about to start on the calling thread, whose word is `w`
// (pending bits clear): say whether it runs on its slot as usual, after taking
// root ownership, or as a foreign region, as another thread owns slot 0.
//
// Before the thread limit is set there are no descriptors: a region of a
// non-worker then runs as a foreign one, which keeps its nesting in the
// thread's word, as a serialized region would (workers exist only after).
// The acquire of the limit orders the descriptors' allocation before their use.
inline Root::Entry Root::enter(SelfWord w) {
  if (w.slot() || w.ownsRoot())
    return Entry::Own;
  if (!settings.limitAcquire())
    return Entry::Foreign;
  if (tryEnter(w))
    return Entry::Taken;
  warnSecondRoot();
  return Entry::Foreign;
}

// Run `fn` inline as a foreign region of one thread.
void runForeign(Microtask fn, int32_t argc, void **argv) {
  SelfWord::set(SelfWord::current().enterForeign());
  invoke(fn, 0, 0, argc, argv);
  SelfWord::set(SelfWord::current().leaveForeign());
}

// -----------------------------------------------------------------------------
// Fork / join
// -----------------------------------------------------------------------------

Level Team::levelFor(int32_t tid) const {
  Level l;
  l.team = const_cast<Team *>(this);
  l.tid = tid;
  l.nth = nth_;
  l.bar = TeamBarrier::enter(*this, tid);
  return l;
}

// For a level-1 team, split the remaining idle workers into one contiguous
// group per member, which that member then prefers for its level-2 team, so
// that concurrent inner forks do not compete and their caches stay valid.
// Each member's group is kept in its thread descriptor, written before the
// wake-up: called once the workers are known.
void Team::partition() {
  SlotMask spare = pool.idle();
  int n = spare.count();
  int32_t slot = leaderSlot();
  SlotMask rest = workers_;
  for (int t = 0; t < nth_; ++t) {
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

void Team::launch(SlotMask workers, int32_t level, int32_t depth, Microtask fn,
    int32_t argc, void **argv) {
  fn_ = fn;
  argc_ = argc;
  memcpy(argv_, argv, sizeof(void *) * size_t(argc));
  nth_ = 1 + workers.count();
  level_ = level;
  depth_ = depth;
  join_.startRegion();
  workers_ = workers;
  bar_.startRegion();
  if (level_ == 1 && settings.maxActive() > 1)
    partition();

  // Assign and wake the members; post only to those that sleep. The team's
  // workers are written first: members use them to wake each other.
  int32_t tid = 1;
  for (int slot : workers)
    threads[slot].assign(this, tid++);
  pool.wakeSleeping(workers);
}

void Team::join(Thread &leader, uint64_t bar) {
  TeamJoin::wait(*this, leader);
  TeamBarrier::endRegion(*this, bar);
}

void Team::runMember(Thread &self, int32_t slot, int32_t tid) {
  uint64_t done = TeamJoin::doneValue(*this);
  // Continue the leader's nesting: same depth, same active levels.
  RegionStack &regions = self.regions();
  regions.startAt(depth_, level_ - 1);
  regions.pushActive(levelFor(tid));
  invoke(fn_, slot, tid, argc_, argv_);
  regions.pop();
  TeamJoin::finish(*this, self, done);
}

void Thread::fork(
    int32_t gtid, int32_t want, Microtask fn, int32_t argc, void **argv) {
  int32_t limit = settings.limit();
  if (want <= 0)
    want = settings.numThreads(regions_.active());
  if (want > limit)
    want = limit;

  // Serialized: one thread asked for, no active level left, nested too
  // deeply, or pool closed.
  Team *team = nullptr;
  SlotMask workers;
  if (want > 1 && regions_.active() < settings.maxActive() &&
      regions_.canNest() && !pool.isClosed(Order::Acquire)) {
    SlotMask prefer;
    if (regions_.active() == 0) {
      if (gtid != 0)
        fatal("a worker started a parallel region outside of a team");
      team = &rootTeam();
      prefer = settings.limitSlots();
    } else {
      team = &slotTeam(gtid);
      prefer = regions_.topIsActive() ? preferred() : SlotMask();
    }
    workers = pool.recruit(*team, want - 1, prefer);
  }
  if (!workers) {
    regions_.pushSerialized();
    invoke(fn, gtid, 0, argc, argv);
    regions_.pop();
    return;
  }

  // Members start at the leader's depth and level. Run as tid 0, then join.
  team->launch(
      workers, regions_.active() + 1, regions_.depth(), fn, argc, argv);
  regions_.pushActive(team->levelFor(0));
  invoke(fn, gtid, 0, argc, argv);
  uint64_t bar = regions_.activeTop()->bar.passed;
  regions_.pop();
  team->join(*this, bar);
  pool.returnWorkers(workers);
}

// The fork of the calling thread, after its arguments are collected.
void forkCall(Microtask fn, int32_t argc, void **argv) {
  SelfWord w = SelfWord::current();
  int32_t want = w.pending();
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
  threads[w.slot()].fork(w.slot(), want, fn, argc, argv);
  if (entry == Root::Entry::Taken)
    root.leave();
}
