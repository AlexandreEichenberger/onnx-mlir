/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- pompSyncTypes.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: the types of the
// synchronization inside a team, which Team and Thread hold by value, so
// they come first. Their code, which uses Team and Thread, is in pompSync.inc.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Synchronization inside a team: types
// -----------------------------------------------------------------------------

// Defined in pompTypes.inc, after these types.
class Team;
class Thread;
struct Level;

// This file covers the synchronization of the threads of a team while they
// run a region: the join at its end, and the barriers inside it.
//
// A worker also waits outside of any team: in the pool, for a team to
// recruit it (Pool::waitWork, in pompPool.inc). After its done report at
// the join of one region, it waits there for the next region it takes part
// in, which may be of another team. It uses the same spin-then-sleep
// protocol, but on pool words (`go`, `sleeping`), woken by the recruiter.
// That wait is not covered here, and uses none of the types below: it has
// its own code and its own mask of sleepers. It shares with them only the
// thread's post, on which both sleep, and the thread's wake-up line, which
// holds both its `go` and its MemberSync, used in turn.
//
// The rest of this file is about the team's synchronization, in two layers.
//
// What members wait for: two protocols, which differ in who waits.
//   TeamJoin     the end of a region ("Join"). One way: each member other
//                than the leader reports done, and only the leader waits. A
//                member does not wait after its report; it goes back to the
//                pool, whose idle wait (pompPool.inc) is the "release" of
//                the join. Hence a done flag per member, which the leader
//                polls, and no release step.
//   Barriers     a barrier inside the region (__kmpc_barrier, "Barriers").
//                All to all: every member waits until all have arrived, and
//                then all go on in the same region. TeamCentralBarrier, a
//                counter, for small teams; TeamTreeBarrier for large ones.
//
// How they wait: TeamWait ("Waits inside a team"), the one wait that all
// use. Spin, yield, then sleep on the waiter's own post, with its bit set in
// the team's `sleepers`. A protocol stores its flag first, then calls wake()
// for the threads that read that flag, so no wake-up is lost. This is the
// same protocol as the pool's idle wait, but on a word of the team: a team's
// wake-ups look only at the team's members, and a thread is in at most one
// wait at a time, idle or in a team.
//
// Where the state lives. A member is one of the team's nth threads: tid 0,
// the leader, and the workers it recruited, tids 1 .. nth-1. A protocol has
// a team part, and may have a member part:
//   Team part    one copy per team, in Team, shared by the members: what all
//                of them must agree on.
//   Member part  one copy per member, written only by that member, so that
//                the writes are spread over lines of their own. Either
//                private, or read by exactly one other thread:
//     MemberBarrier  in the member's region-stack entry, private: the id
//                    of its current barrier, and its links in the barrier
//                    tree.
//     MemberSync     on the thread's wake-up line, the flags that one other
//                    thread polls: `done`, for the leader at the join, and
//                    TreeFlags, for the parent or children in the tree
//                    barrier. The leader uses none. Per thread rather than
//                    per team, as a thread is a non-leader member of at
//                    most one team at a time.
//
// The parts of each protocol:
//
//                    team part                    member part
//   join             TeamJoin: the unique id of   MemberSync::done: one flag
//                    the current region, which    per member, which the
//                    members store into `done`    leader polls
//   central barrier  TeamCentralBarrier: the      none
//                    counter and epoch, the
//                    barrier itself, written by
//                    every member
//   tree barrier     TeamTreeBarrier: the unique  MemberBarrier (barrier id,
//                    id of the team's last        tree links) and TreeFlags:
//                    barrier, from which the      the barrier itself, one
//                    members' ids continue        writer and one reader per
//                                                 flag
//
// Unique ids. The values that the join and the tree barrier store into flags,
// and wait for, are unique ids: never stored twice, by any team. A team's ids,
// one sequence for its joins and one for its tree barriers, each in flags of
// its own, are seeded with the team's index (below kTeams), and each new id is
// the last one plus kTeams. So every id of a team is its index modulo kTeams,
// and two teams never meet; and a team's ids only grow, so a team never meets
// its own earlier ids either. Flags are never reset, and every wait tests for
// equality: a value left in a flag by an earlier region, barrier or team never
// matches. An id is advanced before use, so the first one is index + kTeams: no
// id is 0, which every flag holds at first.
//
// The central barrier accepts contention on its shared lines, which is cheap
// on a few cores. The tree and the join avoid it: per-member flags on
// distinct lines are read in parallel, where one shared line is written in
// turn.

// The wait primitive of the team's protocols (join, barriers): wait
// until a condition on a flag holds, then sleep if it takes too long. It
// defines no flags of its own; `sleepers_` only records who sleeps, so that
// a waker posts only to those threads. See "Waits inside a team".
class TeamWait {
public:
  // Wait, as `self`, until `cond` holds.
  template <class Cond>
  void until(Thread &self, Cond cond);
  // Wake those of the members in `slots` that sleep in a wait of this team.
  void wake(SlotMask slots);
  // Wake member `slot` if it sleeps in a wait of this team.
  void wakeOne(int32_t slot);

private:
  SlotMask claimSleepers(SlotMask slots);

  // A member's slot is in while it sleeps in a wait, or is about to. Written
  // only by sleepers and their wakers, so it stays shared.
  alignas(kCacheLine) AtomicMask sleepers_;
};

// The end of a region: members report, the leader waits. Unlike a barrier,
// the members do not wait for one another, or for the leader. Each member
// other than the leader stores its done flag (MemberSync::done) and leaves
// the team. The leader polls the flags one by one: loads of distinct lines
// overlap, where increments of one counter would serialize. The value is
// the unique id of the region's join (see "Unique ids"), so a flag left by
// an earlier region, or by another team, never matches.
class TeamJoin {
public:
  // At allocation: seed the ids with the team's index.
  void seed(int32_t index) { currUniqueJoinId_ = uint64_t(index); }
  // Leader, before waking the members: the next unique id, for this
  // region's join.
  void startRegion();
  // Member, on entry: the value its done flag takes at the end of the region.
  // Read on entry, with the other fields a member reads then, on the same
  // line, so that the end of the region reads nothing of the team.
  static uint64_t doneValue(const Team &team);
  // Member, at the end of the region: store `done`, from doneValue, into its
  // done flag, and wake the leader. The member's last access to the team.
  static void finish(Team &team, Thread &self, uint64_t done);
  // Leader: wait until every member reported done.
  static void wait(Team &team, Thread &leader);

private:
  // The unique id of the current region's join, which members store into
  // their done flags.
  uint64_t currUniqueJoinId_ = 0;
};

// A barrier inside a region. Unlike the join, every member waits, and all of
// them continue in the region. Two algorithms, each with its own state; a
// team uses one of them, by its size (Team::usesTreeBarriers), and
// Team::barrier picks it. The design of both, and their code, are in
// "Barriers" in pompSync.inc.

// Central barrier, for small teams: the team's part only, in the team.
// Members keep no state of their own for it.
class TeamCentralBarrier {
public:
  // A barrier, for the member at `top`, running on `self`.
  void wait(Team &team, Thread &self, const Level &top);

private:
  // On lines of their own, as every member writes them. `count_` is the
  // number of members that arrived at the current barrier, 0 between
  // barriers (reset by the last to arrive before bumping the epoch), and so
  // at the start of every region, with no reset there. `epoch_`
  // is bumped once per barrier, by the last to arrive, and the others wait for
  // it to move.
  alignas(kCacheLine) Atomic<int> count_;
  alignas(kCacheLine) Atomic<uint64_t> epoch_;
};

// A member's barrier state for a region, in its region-stack entry, set on
// entry (Team::initMemberBarrier). Written only by the member, which keeps it
// private. Only the tree barrier uses it; a member of a team that uses the
// central barrier keeps the default.
struct MemberBarrier {
  // The unique id of this member's current barrier, or of its last one:
  // the team's last on entry, advanced by kTeams at each barrier. The same
  // for every member, as all pass the same barriers.
  uint64_t currUniqueBarrierId = 0;
  // The slots of this member's children (none for a leaf), and of its
  // parent (unused for tid 0). Slots, not tids, so that a barrier reaches
  // their flags directly.
  SlotMask children;
  int32_t parent = 0;
};

// Tree barrier: a member's flags, on its wake-up line (see MemberSync):
// `arrive` is written by the member and polled by its parent, `release` the
// other way. Each holds the value of the last barrier it took part in;
// never reset.
struct TreeFlags {
  Atomic<uint64_t> arrive;
  Atomic<uint64_t> release;
};

// Tree barrier: the team's part, in the team. Only the unique id of the
// team's last barrier, from which the members' ids continue, read once by
// each member on entry; so not aligned, unlike the central barrier: it stays
// on the line of the other fields members read on entry (see Team).
class TeamTreeBarrier {
public:
  // At allocation: seed the ids with the team's index.
  void seed(int32_t index) { lastUniqueBarrierId_ = uint64_t(index); }
  // Member `tid` of `team`, on entry: its part of the barrier.
  MemberBarrier enter(const Team &team, int32_t tid) const;
  // A barrier, for the member at `top`, running on `self`.
  static void wait(Team &team, Thread &self, Level &top);
  // Leader, after the join: `currUniqueBarrierId` is the id of the leader's
  // last barrier, and so every member's, from which the next region of the
  // team continues.
  void endRegion(uint64_t currUniqueBarrierId);

private:
  // The unique id of the team's last tree barrier, in any region.
  uint64_t lastUniqueBarrierId_ = 0;
};
static_assert(sizeof(TeamTreeBarrier) == sizeof(uint64_t),
    "takes the place of a plain counter in Team");

// Synchronization words of a thread as a member of a team, on the thread's
// wake-up line (see Thread). `done` is written by the member and polled by the
// leader at the join, and `tree` is shared with the member's parent or
// children in the barrier tree, so a line is shared by two threads at a time.
// Only members other than the leader use their line: the leader of a team
// stores no `done`, has no parent in the tree, and only reads and writes the
// lines of other members. A thread is such a member of at most one team at
// a time, as it leads any team it forks, and a level-1 member cannot be
// recruited at level 2. So each thread has one line, which it takes from team
// to team. The values stored are unique ids (see "Unique ids"), and waits
// test for equality, so that a value left by a previous team never matches;
// nothing needs to be reset.
struct MemberSync {
  Atomic<uint64_t> done;
  TreeFlags tree;
};
