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
//   TeamBarrier  a barrier inside the region (__kmpc_barrier, "Barriers").
//                All to all: every member waits until all have arrived, and
//                then all go on in the same region. Central counter for
//                small teams, tree for large ones.
//
// How they wait: TeamWait ("Waits inside a team"), the one wait that both
// use. Spin, yield, then sleep on the waiter's own post, with its bit set in
// the team's `sleepers`. A protocol stores its flag first, then calls wake()
// for the threads that read that flag, so no wake-up is lost. This is the
// same protocol as the pool's idle wait, but on a word of the team: a team's
// wake-ups look only at the team's members, and a thread is in at most one
// wait at a time, idle or in a team.
//
// Where the state lives. A member is one of the team's nth threads: tid 0,
// the leader, and the workers it recruited, tids 1 .. nth-1. Each protocol
// has a team part and a member part, and needs both:
//   Team part    one copy per team, in Team, shared by the members: what all
//                of them must agree on.
//   Member part  one copy per member, written only by that member, so that
//                the writes are spread over lines of their own. Either
//                private, or read by exactly one other thread:
//     MemberBarrier  in the member's region-stack entry, private: its count
//                    of barriers passed, and its links in the barrier tree.
//     MemberSync     on the thread's wake-up line, the flags that one other
//                    thread polls: `done`, for the leader at the join, and
//                    TreeFlags, for the parent or children in the tree
//                    barrier. The leader uses none. Per thread rather than
//                    per team, as a thread is a non-leader member of at
//                    most one team at a time.
//
// The two parts of each protocol:
//
//                    team part                    member part
//   join             TeamJoin::region_: tags      MemberSync::done: one flag
//                    `done`, so that a value of   per member, which the
//                    an earlier region never      leader polls
//                    matches
//   central barrier  TeamBarrier::count_ and      MemberBarrier::passed: the
//                    epoch_: the barrier itself,  member's barrier count,
//                    written by every member      kept but not waited on
//   tree barrier     Team::barSeq_ and the team   MemberBarrier (count, tree
//                    tag: the base of the values  links) and TreeFlags: the
//                    that the flags wait for      barrier itself, one writer
//                                                 and one reader per flag
//
// The central barrier accepts contention on its shared lines, which is cheap
// on a few cores. The tree and the join avoid it: per-member flags on
// distinct lines are read in parallel, where one shared line is written in
// turn.

// The wait primitive of the team's protocols (TeamJoin, TeamBarrier): wait
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
// overlap, where increments of one counter would serialize. The values are
// tagged with the region's number, so a flag left by an earlier region
// never matches.
class TeamJoin {
public:
  // Leader, before waking the members.
  void startRegion() { region_++; }
  // Member, on entry: the value its done flag takes at the end of the region.
  static uint64_t doneValue(const Team &team);
  // Member, at the end of the region: report done, `value`, and wake the
  // leader. The member's last access to the team.
  static void finish(Team &team, Thread &self, uint64_t value);
  // Leader: wait until every member reported done.
  static void wait(Team &team, Thread &leader);

private:
  // Number of the current region, among the team's.
  uint64_t region_ = 0;
};

// A barrier inside a region. Unlike the join, every member waits, and all of
// them continue in the region.
//
// Barrier state, in parts that each stay with the data they are used with:
// TeamBarrier and Team::barSeq_ in the team, MemberBarrier in each member's
// region-stack entry, and TreeFlags on each member's wake-up line. The design
// of both barriers, central and tree, and all of the code on this state, are
// in "Barriers" in pompSync.inc.

// A member's part, in its region-stack entry. Written only by the member,
// which keeps it private: it is the member's own copy of the team's state.
struct MemberBarrier {
  // Barriers this thread passed in this team, all regions: the k-th barrier
  // waits for value k (tagged with the team). The same for every member, as
  // all pass the same barriers.
  uint64_t passed = 0;
  // Tree links, set on entry if the team uses the tree: the slots of this
  // member's children (none for a leaf), and of its parent (unused for tid
  // 0). Slots, not tids, so that a barrier reaches their flags directly.
  SlotMask children;
  int32_t parent = 0;
};

// A member's tree flags, on its wake-up line (see MemberSync): `arrive` is
// written by the member and polled by its parent, `release` the other way.
// Each holds the value of the last barrier it took part in; never reset.
struct TreeFlags {
  Atomic<uint64_t> arrive;
  Atomic<uint64_t> release;
};

// The team's part, in the team.
class TeamBarrier {
public:
  // Whether a team of `nth` threads uses the tree, rather than the central
  // barrier.
  static bool usesTree(int32_t nth);
  // Leader, before waking the members.
  void startRegion();
  // Member `tid` of `team`, on entry: its part of the barrier.
  static MemberBarrier enter(const Team &team, int32_t tid);
  // A barrier, for the member at `top`, running on `self`.
  void wait(Team &team, Thread &self, Level &top);
  // Leader, after the join: `passed` is the leader's count, and so every
  // member's, which the next region of `team` starts from.
  static void endRegion(Team &team, uint64_t passed);

private:
  void central(Team &team, Thread &self, Level &top);
  static void tree(Team &team, Thread &self, Level &top);

  // Central barrier, on lines of their own. `count_` is the number of
  // members that arrived at the current barrier, 0 between barriers;
  // `epoch_` is bumped once per barrier, by the last to arrive, and the
  // others wait for it to move. Unused by teams that use the tree.
  alignas(kCacheLine) Atomic<int> count_;
  alignas(kCacheLine) Atomic<uint64_t> epoch_;
};

// Synchronization words of a thread as a member of a team, on the thread's
// wake-up line (see Thread). `done` is written by the member and polled by the
// leader at the join, and `tree` is shared with the member's parent or
// children in the barrier tree, so a line is shared by two threads at a time.
// Only members other than the leader use their line: the leader of a team
// stores no `done`, has no parent in the tree, and only reads and writes the
// lines of other members. A thread is such a member of at most one team at
// a time, as it leads any team it forks, and a level-1 member cannot be
// recruited at level 2. So each thread has one line, which it takes from team
// to team. The values stored are tagged with the team (see Team::tag), and
// waits test for equality, so that a value left by a previous team never
// matches; nothing needs to be reset.
struct MemberSync {
  Atomic<uint64_t> done;
  TreeFlags tree;
};
