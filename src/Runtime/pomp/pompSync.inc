/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompSync.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: the synchronization
// inside a team, declared in pompSyncTypes.inc. TeamWait (the waits of members
// and their wake-ups), TeamJoin (the end of a region) and TeamBarrier.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Waits inside a team (join, barriers)
//
// The wait primitive that the join and the barriers below both use; it
// decides how to wait, they decide on what.
//
// The same protocol as the idle wait, with the team's `sleepers` word in place
// of `sleeping`, and the bit of the waiter's slot. A thread is in at most one
// wait at a time, so it sleeps on its own slot's post, and every post is
// consumed by the wait that set the bit. A waker publishes what it waits for,
// then calls TeamWait::wake; a seq_cst fence on each side orders the publish
// against the read of `sleepers`, and the bit against the read of the
// condition.
// -----------------------------------------------------------------------------

// Conditions of the waits, as function objects.

// The flag holds exactly `value`.
class FlagIs {
public:
  FlagIs(const Atomic<uint64_t> &flag, uint64_t value)
      : flag_(flag), value_(value) {}
  bool operator()() const { return flag_.load(Order::Acquire) == value_; }

private:
  const Atomic<uint64_t> &flag_;
  uint64_t value_;
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

// Spin, then yield, which lets an oversubscribed waitee run, then sleep after
// POMP_SPIN iterations in all.
template <class Cond>
void TeamWait::until(Thread &self, Cond cond) {
  long i = 0;
  SlotMask me = SlotMask::of(self.slot());
  while (!cond()) {
    if (i++ < settings.spin()) {
      if (i < kYieldSpin)
        relax();
      else
        sched_yield();
      continue;
    }
    sleepers_.fetchSet(me);
    fence(Order::SeqCst);
    if (cond()) {
      // A waker that cleared the bit has posted, or will: consume it.
      if (!(sleepers_.fetchClear(me) & me))
        self.waitPost();
      break;
    }
    self.waitPost();
  }
}

// Claim the members in `slots` that sleep in a team wait. Return the
// sleepers whose bit this call cleared, for the caller to post to.
inline SlotMask TeamWait::claimSleepers(SlotMask slots) {
  fence(Order::SeqCst);
  SlotMask s = sleepers_.load(Order::Relaxed) & slots;
  return s ? sleepers_.fetchClear(s) & s : SlotMask();
}

inline void TeamWait::wake(SlotMask slots) {
  for (int slot : claimSleepers(slots))
    threads[slot].post();
}

inline void TeamWait::wakeOne(int32_t slot) {
  if (claimSleepers(SlotMask::of(slot)))
    threads[slot].post();
}

// -----------------------------------------------------------------------------
// Join
//
// The team calls it at three points of a region:
//   Team::launch      startRegion   number the region
//   Team::runMember   doneValue     on entry, and finish at the end
//   Team::join        wait          the leader waits for every member
// Polling one done flag per member: plain loads of distinct lines overlap,
// where increments of one shared counter serialize. A long wait sleeps; each
// member wakes the leader after its done flag.
// -----------------------------------------------------------------------------

inline uint64_t TeamJoin::doneValue(const Team &team) {
  return team.tag(team.join_.region_);
}

// After the store, the leader may reuse the team; only the wake-up of the
// leader follows, which allows it: the leader is found from the team's
// address, not from its contents. The line keeps `done` until the leader
// reads it: the thread joins no other team before the leader returns it to
// the pool, after the join.
inline void TeamJoin::finish(Team &team, Thread &self, uint64_t value) {
  self.sync().done.store(value, Order::Release);
  team.wait_.wakeOne(team.leaderSlot());
}

inline void TeamJoin::wait(Team &team, Thread &leader) {
  uint64_t done = doneValue(team);
  for (int slot : team.workers_)
    team.wait_.until(leader, FlagIs(threads[slot].sync().done, done));
}

// -----------------------------------------------------------------------------
// Barriers
//
// A barrier holds every member of the team until all of them have reached
// it. The compiler emits one at the end of each worksharing loop without
// nowait, and for `#pragma omp barrier`; each member calls __kmpc_barrier,
// which ends in TeamBarrier::wait.
//
// Two algorithms, chosen by team size (usesTree), the same for every member
// of a team, as all of them see the same size:
//
// - Central, for small teams. One shared counter and one generation number
//   (epoch), in the team, on lines of their own. Each member reads the epoch,
//   then increments the counter. The last to arrive (the one that sees
//   nth - 1) resets the counter, bumps the epoch, and wakes the members that
//   sleep; the others wait until the epoch moves. Simple, but every member
//   writes the same line, so the line moves between cores nth times per
//   barrier: cheap on a few cores, slow on many.
//
// - Tree, from POMP_TREE_BARRIER threads (default 5 on s390x, never
//   elsewhere). The members form a 4-ary tree by tid:
//
//     tid 0 (leader)
//     +-- tid 1 --+-- tids 5, 6, 7, 8
//     +-- tid 2 --+-- tids 9, 10, 11, 12
//     +-- tid 3 --+-- tids 13 .. 16
//     +-- tid 4 --+-- tids 17 .. 20
//
//   tid t is the parent of tids 4t+1 .. 4t+4. A barrier has two sweeps:
//     gather: a member waits until each child has stored the barrier's value
//             into its own `arrive`, then stores its own `arrive` for its
//             parent; this goes up to tid 0, which then knows that all have
//             arrived;
//     release: tid 0 stores the value into each child's `release`; each
//             member, once its `release` holds it, does the same for its own
//             children, down to the leaves.
//   Every flag has one writer and one reader, so no line is contended, and
//   the depth is log4(nth): 3 levels for 64 threads.
//
// State, kept with the data it is used with (see pompSyncTypes.inc):
//   Team::bar_           TeamBarrier: the central counter and epoch;
//   Team::barSeq_        the count of barriers at the end of the last region;
//   Level::bar           MemberBarrier, in the member's region-stack entry:
//                        its count of barriers passed, and its tree links;
//   MemberSync::tree     TreeFlags, on the member's wake-up line: `arrive`
//                        and `release`.
//
// Lifetime, the four points at which the team calls into the barrier:
//   Team::launch      startRegion   leader, before waking the members: reset
//                                   the central counter
//   Team::levelFor    enter         each member, on entry to the region: its
//                                   count, from barSeq_, and its tree links
//   Team::barrier     wait          each member, at each barrier
//   Team::join        endRegion     leader, after the join: barSeq_ = its
//                                   count, for the next region
//
// Values instead of resets. Each member counts the barriers it passed in the
// team (MemberBarrier::passed), and the k-th barrier waits for value k, tagged
// with the team (Team::tag): a tree flag is never reset, it is overwritten by
// the next barrier's value, and a value left by an earlier barrier, or by
// another team that used the same thread, never matches. As all members pass
// the same barriers, their counts agree; the count of the leader at the join
// is the team's, and the next region starts from it, so the tags keep
// growing across regions of the team rather than restarting at 1.
//
// Waiting. Every wait is TeamWait::until: spin, yield, then sleep with the
// waiter's bit set in the team's `sleepers`. Every store that a waiter may
// sleep on is followed by TeamWait::wake for the threads that read it.
// -----------------------------------------------------------------------------

// The choice is by team size only, so that every member of a team makes the
// same one.
inline bool TeamBarrier::usesTree(int32_t nth) {
  return nth >= settings.treeMin();
}

// No member is in a barrier of the team yet: the store needs no ordering
// beyond the wake-up of the members, which publishes it.
inline void TeamBarrier::startRegion() { count_.store(0, Order::Relaxed); }

// Every member passed the same barriers. Written after the join, as members
// read it on entry.
inline void TeamBarrier::endRegion(Team &team, uint64_t passed) {
  team.barSeq_ = passed;
}

// The count starts from the team's at the end of the previous region, the
// same for every member.
//
// Tree links: the barrier needs slots, to reach the flags of other threads,
// but the tree is built on tids. Tid 0 is the leader; tid t > 0 is the t-th
// worker in increasing order of slots, so the children of tid t, tids
// ARITY*t+1 .. ARITY*t+ARITY, are consecutive workers: skip the first
// ARITY*t of `workers`, take the next ARITY (fewer at the end of the team).
// The parent of tid t is tid (t-1)/ARITY. Each member finds them once per
// region, here, so that a barrier does no lookup. Example, 12 threads on
// slots 0 (leader), 3, 5, 6, 8, ..: tid 1 is slot 3, its children are tids
// 5..8, its parent tid 0, slot 0.
inline MemberBarrier TeamBarrier::enter(const Team &team, int32_t tid) {
  MemberBarrier m;
  m.passed = team.barSeq_;
  int32_t nth = team.nth_;
  if (!usesTree(nth))
    return m;
  SlotMask workers = team.workers_;
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

// This member's count of barriers moves first: the barrier waits for value
// `passed` in both algorithms (the central one only through the epoch).
void TeamBarrier::wait(Team &team, Thread &self, Level &top) {
  top.bar.passed++;
  if (usesTree(top.nth))
    tree(team, self, top);
  else
    central(team, self, top);
}

// Central barrier: the last member to arrive resets the count and bumps the
// generation, which the others wait for.
//
// The epoch is read before the increment: a member that arrives early reads
// this barrier's epoch, and cannot see the next one, which needs its own
// arrival. The reset of the count comes before the bump of the epoch, which
// publishes it (release), so a member that leaves and arrives at the next
// barrier finds the count at 0. The increment is acq_rel so that the last
// member sees the writes of all others before it releases them.
inline void TeamBarrier::central(Team &team, Thread &self, Level &top) {
  uint64_t epoch = epoch_.load(Order::Acquire);
  if (count_.fetchAdd(1, Order::AcqRel) == top.nth - 1) {
    count_.store(0, Order::Relaxed);
    epoch_.store(epoch + 1, Order::Release);
    team.wait_.wake(team.members() - SlotMask::of(self.slot()));
  } else {
    team.wait_.until(self, FlagLeaves(epoch_, epoch));
  }
}

// Tree barrier: each thread waits for its children to arrive, reports its own
// arrival to its parent and waits for the parent's release, then releases its
// children. Every flag is written by one thread and read by one other, so no
// line is contended; the depth is log_ARITY(nth).
//
// Which flags: a member's `arrive` and `release` are on its own line, so a
// thread writes its own `arrive` and reads its own `release`, and reads and
// writes the other ones on its children's lines.
//
// Ordering: a store with release, then a read of the same flag with acquire,
// carries everything the writer did before the barrier to the reader. The
// gather carries it up to tid 0, which has then seen all members' work, and
// the release sweep carries it down to every member.
//
// Waits test for equality, as lines move from team to team (see MemberSync).
// No flag can pass the value its reader waits for: a child publishes the next
// `arrive` only after this barrier's `release`, which its parent sets only
// after this barrier's `arrive`.
void TeamBarrier::tree(Team &team, Thread &self, Level &top) {
  const MemberBarrier &m = top.bar;
  uint64_t v = team.tag(m.passed);
  // Gather: all of this member's subtree has arrived.
  for (int c : m.children)
    team.wait_.until(self, FlagIs(threads[c].sync().tree.arrive, v));
  // Tid 0 has no parent: once its children arrived, everyone has.
  if (top.tid != 0) {
    TreeFlags &flags = self.sync().tree;
    flags.arrive.store(v, Order::Release);
    team.wait_.wake(SlotMask::of(m.parent));
    team.wait_.until(self, FlagIs(flags.release, v));
  }
  // Release: pass it down to this member's children.
  for (int c : m.children)
    threads[c].sync().tree.release.store(v, Order::Release);
  if (m.children)
    team.wait_.wake(m.children);
}
