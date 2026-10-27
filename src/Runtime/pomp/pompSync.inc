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
// and their wake-ups), TeamJoin (the end of a region), and the barriers:
// TeamCentralBarrier and TeamTreeBarrier.
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

// -----------------------------------------------------------------------------
// Conditions of the waits, as function objects.

// The flag holds exactly `value`. Save the address (to be polled) and the
// desired value.
class FlagIs {
public:
  FlagIs(const Atomic<uint64_t> &flag, uint64_t value)
      : flag_(flag), value_(value) {}
  bool operator()() const { return flag_.load(Order::Acquire) == value_; }

private:
  const Atomic<uint64_t> &flag_;
  uint64_t value_;
};

// The flag no longer holds `value`,  Save the address (to be polled) and the
// undesired value.
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

// Spin, then yield, which lets an oversubscribed waitee run, then sleep after
// POMP_SPIN iterations in all.
template <class Cond>
void TeamWait::until(Thread &self, Cond cond) {
  long i = 0;
  SlotMask meMask = SlotMask::of(self.slot());
  while (!cond()) {
    if (i++ < settings.spin()) {
      if (i < kYieldSpin)
        relax(); // Initially just relax to avoid yielding too early.
      else
        sched_yield(); // Then yield to let a waitee run, before sleeping.
      continue;
    }
    // Now sleep by (1) setting our bit in the mask; (2) fencing, so that a
    // waker sees the bit or we see its publish; (3) testing cond again, to
    // catch a waker that published before the bit was set (explanation
    // below); (4) if cond still fails, waiting for the post and looping.
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
    self.waitPost();
  }
}

// Claim the members in `slots` that sleep in a team wait. Return the
// sleepers whose bit this call cleared, for the caller to post to.
inline SlotMask TeamWait::claimSleepers(SlotMask slots) {
  fence(Order::SeqCst);
  SlotMask s = sleepers_.load(Order::Relaxed) & slots;
  return s ? sleepers_.clearAndReport(s) : SlotMask();
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
// TeamJoin
//
// The team calls it at three points of a region:
//   Team::launch      startRegion   the region's unique join id
//   Team::runMember   doneValue     on entry, and finish at the end
//   Team::join        wait          the leader waits for every member
// Polling one done flag per member: plain loads of distinct lines overlap,
// where increments of one shared counter serialize. A long wait sleeps; each
// member wakes the leader after its done flag.
// -----------------------------------------------------------------------------

inline void TeamJoin::startRegion() { currUniqueJoinId_ += kTeams; }

inline uint64_t TeamJoin::doneValue(const Team &team) {
  return team.join_.currUniqueJoinId_;
}

// After the store, the leader may reuse the team; only the wake-up of the
// leader follows, which allows it: the leader is found from the team's
// address, not from its contents. The line keeps `done` until the leader
// reads it: the thread joins no other team before the leader returns it to
// the pool, after the join.
inline void TeamJoin::finish(Team &team, Thread &self, uint64_t done) {
  self.sync().done.store(done, Order::Release);
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
// which ends in Team::barrier.
//
// Two algorithms, chosen by team size (Team::usesTreeBarriers), the same for
// every member of a team, as all of them see the same size. Each has its own
// state, below: TeamCentralBarrier, and TeamTreeBarrier with the members'
// MemberBarrier and TreeFlags.
//
// The team calls into the barriers in three phases of a region, in this
// order, and in each calls only the algorithm of its size, which touches only
// its own state:
//                                                     central    tree
//   1. Team::levelFor  initMemberBarrier  each member  -          enter
//   2. __kmpc_barrier  barrier            each member  wait       wait
//   3. Team::join      endLeaderBarrier   leader       -          endRegion
// Phase 1 runs in every member, leader included, unsynchronized: it reads
// team fields that the wake-up published, and writes the member's own state.
// Phase 3 runs after the join, which orders it after every member's last
// barrier.
//
// Waiting. Every wait is TeamWait::until: spin, yield, then sleep with the
// waiter's bit set in the team's `sleepers`. Every store that a waiter may
// sleep on is followed by TeamWait::wake for the threads that read it.
// -----------------------------------------------------------------------------

// The choice is by team size only, so that every member of a team makes the
// same one.
inline bool Team::usesTreeBarriers(int32_t nth) {
  return nth >= settings.treeMin();
}

// A team that uses the central barrier needs no member state: the default.
inline MemberBarrier Team::initMemberBarrier(int32_t tid) const {
  return usesTreeBarriers(nth_) ? tree_.enter(*this, tid) : MemberBarrier();
}

inline void Team::barrier(Thread &self, Level &top) {
  if (usesTreeBarriers(top.nth))
    TeamTreeBarrier::wait(*this, self, top);
  else
    central_.wait(*this, self, top);
}

inline void Team::endLeaderBarrier(uint64_t currUniqueBarrierId) {
  if (usesTreeBarriers(nth_))
    tree_.endRegion(currUniqueBarrierId);
}

// Both sequences, from the same seed: they are stored into different flags,
// `done` and the tree flags, so they need not differ from each other.
inline void Team::seedUniqueIds(int32_t index) {
  join_.seed(index);
  tree_.seed(index);
}

// -----------------------------------------------------------------------------
// Central barrier
//
// For small teams. One shared counter and one generation number (epoch), in
// the team, on lines of their own. Each member reads the epoch, then
// increments the counter. The last to arrive (the one that sees nth - 1)
// resets the counter, bumps the epoch, and wakes the members that sleep; the
// others wait until the epoch moves. Simple, but every member writes the
// same line, so the line moves between cores nth times per barrier: cheap
// on a few cores, slow on many.
//
// State: TeamCentralBarrier, in the team (Team::central_); none in the
// members, as the epoch, read on arrival, tells each member which barrier
// it waits for.
//
// Lifetime, for a team that uses it: only phase 2, `wait`. No setup: the
// counter is 0 between barriers, as the last to arrive resets it, and so at
// the start of every region, as a region ends only after every member has
// left its last barrier. It starts at 0 too, as the descriptors are zeroed.
// -----------------------------------------------------------------------------

// The last member to arrive resets the count and bumps the generation, which
// the others wait for.
//
// The epoch is read before the increment: a member that arrives early reads
// this barrier's epoch, and cannot see the next one, which needs its own
// arrival. The reset of the count comes before the bump of the epoch, which
// publishes it (release), so a member that leaves and arrives at the next
// barrier finds the count at 0. The increment is acq_rel so that the last
// member sees the writes of all others before it releases them.
inline void TeamCentralBarrier::wait(
    Team &team, Thread &self, const Level &top) {
  uint64_t epoch = epoch_.load(Order::Acquire);
  if (count_.fetchAdd(1, Order::AcqRel) == top.nth - 1) {
    // Last to arrive: reset the count, bump the epoch, and wake the members.
    count_.store(0, Order::Relaxed);
    epoch_.store(epoch + 1, Order::Release);
    team.wait_.wake(team.members() - SlotMask::of(self.slot()));
  } else {
    // Not last: wait for the epoch to move, which the last member does after.
    team.wait_.until(self, FlagLeaves(epoch_, epoch));
  }
}

// -----------------------------------------------------------------------------
// Tree barrier
//
// From POMP_TREE_BARRIER threads (default 5 on s390x, never elsewhere). The
// members form a 4-ary tree by tid:
//
//   tid 0 (leader)
//   +-- tid 1 --+-- tids 5, 6, 7, 8
//   +-- tid 2 --+-- tids 9, 10, 11, 12
//   +-- tid 3 --+-- tids 13 .. 16
//   +-- tid 4 --+-- tids 17 .. 20
//
// tid t is the parent of tids 4t+1 .. 4t+4. A barrier has two sweeps:
//   gather:  a member waits until each child has stored the barrier's value
//            into its own `arrive`, then stores its own `arrive` for its
//            parent; this goes up to tid 0, which then knows that all have
//            arrived;
//   release: tid 0 stores the value into each child's `release`; each
//            member, once its `release` holds it, does the same for its own
//            children, down to the leaves.
// Every flag has one writer and one reader, so no line is contended, and the
// depth is log4(nth): 3 levels for 64 threads.
//
// State, kept with the data it is used with (see pompSyncTypes.inc):
//   Team::tree_          TeamTreeBarrier: the unique id of the team's last
//                        barrier, in any region;
//   Level::bar           MemberBarrier, in the member's region-stack
//                        entry: the unique id of its current barrier, and
//                        its tree links;
//   MemberSync::tree     TreeFlags, on the member's wake-up line: `arrive`
//                        and `release`.
//
// Lifetime, the three phases, for a team that uses it:
//   1. enter       each member, on entry to the region: its barrier id, the
//                  team's last, and its tree links
//   2. wait        each member, at each barrier: the next id, then the
//                  barrier on it
//   3. endRegion   leader, after the join: the team's last id = its own, for
//                  the next region
//
// Values instead of resets. Each barrier waits for its unique id (see
// "Unique ids" in pompSyncTypes.inc): a flag is never reset, it is
// overwritten by the next barrier's id, and a value left by an earlier
// barrier, or by another team that used the same thread, never matches.
// Each member keeps the id in its own entry (currUniqueBarrierId), starts
// from the team's last on entry, and advances it by kTeams at each barrier;
// as all members pass the same barriers, their ids agree. The leader's id at
// the join is the team's last, from which the next region continues, so the
// ids keep growing across regions of the team rather than restarting. A
// region in which the team uses the central barrier touches none of this:
// the team's last id waits, unchanged, for the next region that uses the
// tree.
//
// The team's last id only grows, so that a flag never sees a value twice: a
// team's size may change from region to region, and with it the algorithm,
// but the team's id is never lowered.
// -----------------------------------------------------------------------------

// The id starts from the team's last, the same for every member.
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
inline MemberBarrier TeamTreeBarrier::enter(
    const Team &team, int32_t tid) const {
  MemberBarrier m;
  m.currUniqueBarrierId = lastUniqueBarrierId_;
  int32_t nth = team.nth_;
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

// Every member passed the same barriers. Written after the join, as members
// read it on entry.
inline void TeamTreeBarrier::endRegion(uint64_t currUniqueBarrierId) {
  lastUniqueBarrierId_ = currUniqueBarrierId;
}

// Each thread waits for its children to arrive, reports its own arrival to
// its parent and waits for the parent's release, then releases its children.
// Every flag is written by one thread and read by one other, so no line is
// contended; the depth is log_ARITY(nth).
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
void TeamTreeBarrier::wait(Team &team, Thread &self, Level &top) {
  MemberBarrier &m = top.bar;
  uint64_t v = m.currUniqueBarrierId += kTeams;
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
