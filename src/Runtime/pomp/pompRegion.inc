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

inline Thread *Thread::current() { return Self::get(); }

inline Thread &Thread::currentOrNewRoot() {
  Thread *self = Self::get();
  return self ? *self : roots.acquire();
}

inline RegionStack *Thread::currentRegions() {
  Thread *self = current();
  return self ? &self->regions() : nullptr;
}

inline bool Thread::callerInPomp() {
  Thread *self = current();
  return self && (self->slot() != 0 || self->regions().depth() > 0);
}

// -----------------------------------------------------------------------------
// Root blocks
//
// A thread that is not in the pool becomes a root at its first region (or
// pushed team size): it takes a root block, its descriptor and the two teams
// it leads, its own until it ends. So roots never share a descriptor, any
// number of them may run regions at once, and the next region of a root
// touches nothing global but the pool.
//
// When the thread ends, the destructor of `exitKey_` returns the block to the
// free list, and a later root may take it. Blocks are never freed.
//
// Why: a late wake-up. At the end of a region each worker stores its done
// flag, then wakes the leader (TeamJoin::workerFinish). Once the last flag is
// stored the leader may join, return, and its thread end; a worker may be
// delayed for any time between its store and its wake-up, so that wake-up
// may come while the block is on the free list or owned by another root. It
// touches, in the block:
//   - `sleepers_` of the team: one load, and an atomic clear of bit 0 if it
//     is set;
//   - the leader Thread's `post_` (mutex, condition, flag), only if it
//     cleared that bit.
// It is harmless to a new owner: bit 0 is set only while the owner waits in
// that team (TeamWait::until always leaves with its bit clear), and a waker
// that clears a sleeper's bit owes it one post, which this one makes; a post
// only means "re-check". A late done flag holds an old join id, which no
// later join waits for.
//
// So, for any change here or in the team waits:
//   1. Never free a block, nor its Thread or teams: the late wake-up would
//      use freed memory, a pthread mutex among it.
//   2. Never initialize a block's post again (Thread::init does it once): a
//      late waker may still be inside its post().
//   3. Never reset a team's unique ids on reuse: they must keep growing, or a
//      late or stale flag could match a new region's id. A block keeps its
//      team indices too. Roots::acquire resets only the pending team size.
//   4. Recycle a block whole: its teams' `leader_` must stay its Thread, as
//      a late wake-up of team T posts to T's leader_.
//   5. After its done store, a worker reads nothing of the team but fields
//      set at allocation and never changed (`leader_`, read before the
//      store, and the leader's slot): any other would belong to the region
//      of the block's next owner.
// test/pomp_host_test.c latepost (S18) stresses this case.
// -----------------------------------------------------------------------------

// The destructor of Roots::exitKey_, with the block of the ending thread.
void recycleRootBlock(void *block) {
  roots.recycle(*static_cast<RootBlock *>(block));
}

void Roots::init() {
  mutex_.init();
  checkPthread(
      pthread_key_create(&exitKey_, recycleRootBlock), "pthread_key_create");
}

RootBlock *Roots::allocate() {
  // Team indices: below kMaxThreads for the pool's teams, then two per block.
  int64_t index = int64_t(kMaxThreads) + 2 * int64_t(count_);
  if (uint64_t(index + 1) >= kIdStride)
    fatal("too many host threads running OpenMP regions at once");
  RootBlock *b = static_cast<RootBlock *>(allocLines(1, sizeof(RootBlock)));
  if (!b)
    fatal("cannot allocate the descriptor of a host thread");
  b->thread.init();
  b->thread.initIdentity(0, &b->levelOne, &b->levelTwo);
  b->levelOne.init(int32_t(index), b->thread);
  b->levelTwo.init(int32_t(index + 1), b->thread);
  ++count_;
  POMP_STAT(kStatRootBlocks);
  return b;
}

// A reused block keeps all of its state but the pending team size: see rules
// 2 and 3 of "Root blocks".
Thread &Roots::acquire() {
  RootBlock *b;
  {
    ScopedLock<Mutex> lock(mutex_);
    b = free_;
    if (b)
      free_ = b->next;
    else
      b = allocate();
  }
  b->thread.setPending(0);
  checkPthread(pthread_setspecific(exitKey_, b), "pthread_setspecific");
  Self::set(&b->thread);
  return b->thread;
}

// The value of exitKey_ is already null here: use `block`, not Self. A thread
// that ends inside a region (pthread_exit) keeps its block, never reused: its
// team may still use it.
void Roots::recycle(RootBlock &block) {
  Self::set(nullptr);
  if (block.thread.regions().depth() != 0)
    return;
  ScopedLock<Mutex> lock(mutex_);
  block.next = free_;
  free_ = &block;
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
