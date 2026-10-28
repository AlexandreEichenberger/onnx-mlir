/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- pompSyncTypes.inc --------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the types of the synchronization inside a team. Team and
// Thread hold them by value, so they come first; their code is in pompSync.inc.
//
//===----------------------------------------------------------------------===//

// Defined in pompTypes.inc, after these types.
class Team;
class Thread;
struct Level;

// Two protocols run inside a region, both waiting through TeamWait:
//   TeamJoin   end of the region: workers report done, only the leader waits.
//   Barriers   __kmpc_barrier: all wait for all. TeamCentralBarrier (one
//              counter) for small teams, TeamTreeBarrier for large ones.
// Each keeps a team part, in Team, and may keep a member part: MemberBarrier
// (private, in the region-stack entry) and WorkerSync (flags polled by one
// other thread, on the thread's wake-up line). Roles: see pompIdentity.inc.
// The idle wait of the pool
// (Pool::waitWork) is separate and uses none of these types.

// -----------------------------------------------------------------------------
// UniqueId and UniqueId Flag
// -----------------------------------------------------------------------------

// Unique ids. The join and the tree barrier store and wait for ids that are
// never stored twice, by any team: a team's ids start at its index (below
// kIdStride) and grow by kIdStride, so they are its index modulo kIdStride
// and only grow. Flags are never reset and waits test for equality, so a value
// left by an earlier region, barrier or team never matches. Ids advance before
// use, so none is 0, the initial value of every flag.
//
// UniqueId and UniqueIdFlag enforce this: an id is made only by seed, changes
// only by advance, and is only compared for equality; a flag holds only ids,
// stored with release and tested with acquire. Zero bits are the unseeded id,
// so both are valid in the zero-allocated descriptors.
class UniqueId {
public:
  constexpr UniqueId() = default;
  // The first id of team `index`, in [0, kIdStride).
  static UniqueId seed(int32_t index) {
    if (uint64_t(uint32_t(index)) >= kIdStride)
      fatal("unique id seed out of range");
    UniqueId id;
    id.v_ = uint64_t(index);
    return id;
  }
  // Move to the next id of the same team; return it.
  UniqueId advance() {
    v_ += kIdStride;
    return *this;
  }
  bool operator==(UniqueId other) const { return v_ == other.v_; }

private:
  friend class UniqueIdFlag;
  uint64_t v_ = 0;
};

// A flag that one thread stores ids into and another waits on. Never reset.
class UniqueIdFlag {
public:
  void store(UniqueId id) { flag_.store(id.v_, Order::Release); }
  bool is(UniqueId id) const { return flag_.load(Order::Acquire) == id.v_; }

private:
  Atomic<uint64_t> flag_;
};
static_assert(sizeof(UniqueId) == 8 && __is_trivially_copyable(UniqueId),
    "passed and copied as a plain 64-bit value");

// -----------------------------------------------------------------------------
// Team Wait and Join
// -----------------------------------------------------------------------------

// The wait of the join and barriers: spin until a condition holds, then
// sleep. `sleepers_` records who sleeps, so wakers post only to them.
class TeamWait {
public:
  // Wait, as `self`, until `cond` holds.
  template <class Cond>
  void until(Thread &self, Cond cond);
  // Wake those of the members of `team` in `slots` that sleep in a wait of
  // this team (`team`'s own).
  void wake(const Team &team, SlotMask slots);
  // Wake member `t` if it sleeps in a wait of this team.
  void wakeOne(Thread &t);

private:
  // Clear the bits of the sleepers among `slots`; return those it cleared.
  SlotMask claimSleepers(SlotMask slots);

  // Slots of members sleeping in a wait, or about to.
  alignas(kCacheLine) AtomicMask sleepers_;
};

// The end of a region: each worker stores the join's unique id into its
// WorkerSync::done and leaves, without waiting; the leader polls the flags.
class TeamJoin {
public:
  // At allocation: seed the ids with the team's index.
  void seed(int32_t index) { currUniqueJoinId_ = UniqueId::seed(index); }
  // Before the wake-up: advance to this region's join id.
  void leaderStart();
  // On entry: the value of the worker's done flag at the end. Read with the
  // other entry fields, so the end of the region reads nothing of the team.
  static UniqueId workerDoneValue(const Team &team);
  // At the end: store `done` and wake the leader. The worker's last access to
  // the team.
  static void workerFinish(Team &team, Thread &self, UniqueId done);
  // Wait until every worker reported done.
  static void leaderWait(Team &team, Thread &leader);

private:
  // The current region's join id.
  UniqueId currUniqueJoinId_;
};

// -----------------------------------------------------------------------------
// Barriers
// -----------------------------------------------------------------------------

// Barriers: one of two algorithms per team, by size (Team::usesTreeBarriers).
// See "Barriers" in pompSync.inc.

// Central barrier, for small teams. No member state.
class TeamCentralBarrier {
public:
  // A barrier, for the member at `innermost`, running on `self`.
  void wait(Team &team, Thread &self, const Level &innermost);

private:
  // Each on its own line, as every member writes them. `count_`: members
  // arrived, 0 between barriers. `epoch_`: bumped once per barrier.
  alignas(kCacheLine) Atomic<int> count_;
  alignas(kCacheLine) Atomic<uint64_t> epoch_;
};

// A member's private barrier state, in its region-stack entry, set on entry.
// Used by the tree barrier only.
struct MemberBarrier {
  // Id of the current (or last) barrier; the same for every member.
  UniqueId currUniqueBarrierId;
  // Slots, not tids, of the tree children; the parent (unused for tid 0),
  // which may be the leader, so not a slot.
  SlotMask children;
  Thread *parent = nullptr;
};

// Tree barrier flags of a member: `arrive` is written by the member and
// polled by its parent, `release` the other way.
struct TreeFlags {
  UniqueIdFlag arrive;
  UniqueIdFlag release;
};

// Tree barrier, team part: the id of the team's last barrier, read by each
// member on entry. Not aligned: it shares the line of the other entry fields.
class TeamTreeBarrier {
public:
  // At allocation: seed the ids with the team's index.
  void seed(int32_t index) { lastUniqueBarrierId_ = UniqueId::seed(index); }
  // Member `tid` of `team`, on entry: its part of the barrier.
  MemberBarrier enter(const Team &team, int32_t tid) const;
  // A barrier, for the member at `innermost`, running on `self`.
  static void wait(Team &team, Thread &self, Level &innermost);
  // After the join: the next region continues from `currUniqueBarrierId`,
  // the last barrier id of the region.
  void leaderEndRegion(UniqueId currUniqueBarrierId);

private:
  UniqueId lastUniqueBarrierId_;
};
static_assert(sizeof(TeamTreeBarrier) == sizeof(UniqueId),
    "takes the place of a plain counter in Team");

// A thread's flags as a worker in a team, on its wake-up line: `done`, polled
// by the leader, and `tree`, shared with its tree parent or children. Per
// thread, not per team: a thread is a worker of at most one team at a time
// (it leads any team it forks). Never reset; see "Unique ids".
struct WorkerSync {
  UniqueIdFlag done;
  TreeFlags tree;
};
