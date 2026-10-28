/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompSync.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the synchronization inside a team, declared in
// pompSyncTypes.inc: TeamWait, TeamJoin, and the barriers.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Waits inside a team (join, barriers)
//
// TeamWait decides how to wait; the join and barriers decide on what. The
// same protocol as the idle wait, on the team's `sleepers`: a waker publishes,
// then calls wake(). A thread is in at most one wait at a time, so it sleeps
// on its own post.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Conditions of the waits.

// The flag holds unique id `value`.
class FlagIs {
public:
  FlagIs(const UniqueIdFlag &flag, UniqueId value)
      : flag_(flag), value_(value) {}
  bool operator()() const { return flag_.is(value_); }

private:
  const UniqueIdFlag &flag_;
  UniqueId value_;
};

// The flag no longer holds `value`.
class FlagLeaves {
public:
  FlagLeaves(const Atomic<uint64_t> &flag, uint64_t value)
      : flag_(flag), value_(value) {}
  bool operator()() const { return flag_.load(Order::Acquire) != value_; }

private:
  const Atomic<uint64_t> &flag_;
  uint64_t value_;
};

// -----------------------------------------------------------------------------
// TeamWait

// Spin, then sleep: for POMP_SPIN us, or under auto for this thread's
// adaptive team-wait spin, updated with the length of this wait. No yield: a
// waiting thread that yields still slows an SMT sibling that computes, and
// an oversubscribed waitee gets its CPU back once the waiter sleeps.
template <class Cond>
void TeamWait::until(Thread &self, Cond cond) {
  bool adapt = settings.spinAdaptive();
  long maxUs = settings.spinMaxUs();
  // A wait that is already over costs no clock read.
  if (cond())
    return;
  uint64_t t0 = adapt ? nowUs() : 0;
  struct Update {
    bool adapt;
    long maxUs;
    uint64_t t0;
    AdaptiveSpin &spin;
    ~Update() {
      if (adapt)
        spin.update(nowUs() - t0, maxUs);
    }
  } update{adapt, maxUs, t0, self.teamSpin_};
  SpinBudget spin(adapt ? self.teamSpin_.next(maxUs) : settings.spinUs());
  SlotMask meMask = SlotMask::of(self.slot());
  while (!cond()) {
    if (spin.more()) {
      relax();
      continue;
    }
    // Sleep: set our bit, fence (pairs with the waker's, in claimSleepers:
    // either it sees the bit or we see its publish), and re-check.
    POMP_STAT(kStatTeamSleeps);
    sleepers_.fetchSet(meMask);
    fence(Order::SeqCst);
    if (cond()) {
      // Leave the sleepers. If a waker cleared our bit first, it has posted
      // or will: wait for that post, so that it does not stay pending and
      // end a later sleep of this thread.
      if (!sleepers_.clearAndReport(meMask))
        self.waitPost();
      break;
    }
    // Woken, or a stale post: re-check.
    self.waitPost();
  }
}

// The sleepers among `slots` whose bit this call cleared, to post to.
inline SlotMask TeamWait::claimSleepers(SlotMask slots) {
  fence(Order::SeqCst);
  SlotMask s = sleepers_.load(Order::Relaxed) & slots;
  return s ? sleepers_.clearAndReport(s) : SlotMask();
}

inline void TeamWait::wake(SlotMask slots) {
  for (int32_t slot : claimSleepers(slots))
    threads[slot].post();
}

inline void TeamWait::wakeOne(int32_t slot) {
  if (claimSleepers(SlotMask::of(slot)))
    threads[slot].post();
}

// -----------------------------------------------------------------------------
// TeamJoin
//
// Polls one done flag per worker: loads of distinct lines overlap, where
// increments of one counter serialize.
// -----------------------------------------------------------------------------

inline void TeamJoin::leaderStart() { currUniqueJoinId_.advance(); }

inline UniqueId TeamJoin::workerDoneValue(const Team &team) {
  return team.join_.currUniqueJoinId_;
}

// After the store the leader may reuse the team: the wake-up that follows
// uses only the team's address.
inline void TeamJoin::workerFinish(Team &team, Thread &self, UniqueId done) {
  self.sync().done.store(done);
  team.wait_.wakeOne(team.leaderSlot());
}

inline void TeamJoin::leaderWait(Team &team, Thread &leader) {
  UniqueId done = team.join_.currUniqueJoinId_;
  for (int32_t slot : team.workers_)
    team.wait_.until(leader, FlagIs(threads[slot].sync().done, done));
}

// -----------------------------------------------------------------------------
// Barriers
//
// One of two algorithms per team, by size, so every member picks the same.
// Per region: initBarrier on entry (each member), barrier (each
// member, at each __kmpc_barrier), leaderEndBarrier after the join (leader).
// -----------------------------------------------------------------------------

inline bool Team::usesTreeBarriers(int32_t nth) {
  return nth >= settings.treeMin();
}

inline MemberBarrier Team::initBarrier(int32_t tid) const {
  return usesTreeBarriers(nth_) ? tree_.enter(*this, tid) : MemberBarrier();
}

inline void Team::barrier(Thread &self, Level &innermost) {
  if (usesTreeBarriers(innermost.nth))
    TeamTreeBarrier::wait(*this, self, innermost);
  else
    central_.wait(*this, self, innermost);
}

inline void Team::leaderEndBarrier(UniqueId currUniqueBarrierId) {
  if (usesTreeBarriers(nth_))
    tree_.leaderEndRegion(currUniqueBarrierId);
}

// Join and tree ids share a seed: they go to different flags.
inline void Team::init() {
  int32_t i = index();
  join_.seed(i);
  tree_.seed(i);
}

// -----------------------------------------------------------------------------
// Central barrier
//
// For small teams: one shared counter and epoch. Every member writes the same
// line, so it is cheap on a few cores, slow on many.
// -----------------------------------------------------------------------------

inline void TeamCentralBarrier::wait(
    Team &team, Thread &self, const Level &innermost) {
  // Read before arriving: the next barrier's epoch needs our own arrival.
  uint64_t epoch = epoch_.load(Order::Acquire);
  // Acq_rel: the last to arrive sees every member's writes.
  if (count_.fetchAdd(1, Order::AcqRel) == innermost.nth - 1) {
    // Last: reset the count, then bump the epoch, which publishes the reset
    // to the next barrier, and wake the members.
    count_.store(0, Order::Relaxed);
    epoch_.store(epoch + 1, Order::Release);
    team.wait_.wake(team.members() - SlotMask::of(self.slot()));
  } else {
    team.wait_.until(self, FlagLeaves(epoch_, epoch));
  }
}

// -----------------------------------------------------------------------------
// Tree barrier
//
// From POMP_TREE_BARRIER threads (default 5 on s390x, never elsewhere). The
// members form a 4-ary tree by tid, tid t the parent of 4t+1 .. 4t+4:
//
//   tid 0 (leader)
//   +-- tid 1 --+-- tids 5, 6, 7, 8
//   +-- tid 2 --+-- tids 9, 10, 11, 12
//   +-- tid 3 --+-- tids 13 .. 16
//   +-- tid 4 --+-- tids 17 .. 20
//
// Arrivals gather up to tid 0, releases go back down. Every flag has one
// writer and one reader, so no line is contended; depth is log4(nth).
// -----------------------------------------------------------------------------

// Once per region, so that a barrier does no lookup: the starting id and
// the tree links, as slots.
inline MemberBarrier TeamTreeBarrier::enter(
    const Team &team, int32_t tid) const {
  MemberBarrier m;
  // Continue from the team's last id, so ids keep growing across regions.
  m.currUniqueBarrierId = lastUniqueBarrierId_;
  int32_t nth = team.nth_;
  SlotMask workers = team.workers_;
  // Tid t > 0 is the t-th worker by slot, so the children, tids
  // ARITY*t+1 .. ARITY*t+ARITY, are consecutive workers: skip the first
  // ARITY*t, take up to ARITY. E.g. workers on slots 3, 5, 6, 8, ..: tid 1 is
  // slot 3, its children are tids 5..8.
  int32_t first = kTreeArity * tid + 1;
  int32_t n = nth - first;
  if (n > kTreeArity)
    n = kTreeArity;
  if (n > 0)
    m.children = (workers - workers.lowest(first - 1)).lowest(n);
  if (tid != 0) {
    int32_t ptid = (tid - 1) / kTreeArity;
    m.parent = ptid == 0 ? team.leaderSlot()
                         : (workers - workers.lowest(ptid - 1)).lowest();
  }
  return m;
}

// After the join, as members read it on entry. Only grows: a region that uses
// the central barrier does not touch it.
inline void TeamTreeBarrier::leaderEndRegion(UniqueId currUniqueBarrierId) {
  lastUniqueBarrierId_ = currUniqueBarrierId;
}

// Release stores and acquire loads carry every member's work up to tid 0 and
// back down. No flag can pass the value its reader waits for: a child's next
// `arrive` follows this barrier's `release`.
void TeamTreeBarrier::wait(Team &team, Thread &self, Level &innermost) {
  MemberBarrier &m = innermost.bar;
  // This barrier's unique id, the same for every member.
  UniqueId v = m.currUniqueBarrierId.advance();
  // Gather: wait for the children's `arrive`, on their lines.
  for (int32_t c : m.children)
    team.wait_.until(self, FlagIs(threads[c].sync().tree.arrive, v));
  // Then report our own to the parent, and wait for its release. Tid 0 has
  // no parent: once its children arrived, everyone has.
  if (innermost.tid != 0) {
    TreeFlags &flags = self.sync().tree;
    flags.arrive.store(v);
    team.wait_.wake(SlotMask::of(m.parent));
    team.wait_.until(self, FlagIs(flags.release, v));
  }
  // Release: pass it down to the children.
  for (int32_t c : m.children)
    threads[c].sync().tree.release.store(v);
  if (m.children)
    team.wait_.wake(m.children);
}
