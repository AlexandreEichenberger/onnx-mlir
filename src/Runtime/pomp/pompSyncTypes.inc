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

// The team's synchronization is in three components, each a member of Team
// with its code in a section of its own: TeamWait, the waits of members and
// their wake-ups ("Waits inside a team"); TeamJoin, the end of a region
// ("Join"); and TeamBarrier ("Barriers").

// Waits of a team's members, leader included, on flags of the team: spin,
// then sleep on the thread's own post. A waker publishes what it waits for,
// then wakes the waiters it may concern; see "Waits inside a team".
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

// The join: each member other than the leader reports done on its own sync
// line (MemberSync::done), and the leader polls them. The values are tagged
// with the region's number.
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
