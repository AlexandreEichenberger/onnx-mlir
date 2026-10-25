/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompPool.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp, included in its anonymous namespace: the pool of workers.
// The idle wait and its wake-ups, the recruitment of a team's workers, and the
// host protocol: entry, close, release and reopen.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Wake-up and sleep
//
// An idle worker spins on its `go` epoch and on its bit in `released`, then
// sleeps. To sleep, it sets its bit in `sleeping`, re-reads both, and waits for
// a post. A waker first publishes (bumps `go`, or sets the `released` bit),
// then reads `sleeping`, and posts only to workers whose bit it clears itself.
// All these accesses are sequentially consistent, so either the worker sees
// the new value or the waker sees the bit: no wake-up is lost. Exactly one of
// them clears the bit, so every post is consumed in the sleep it was meant for.
// Spinning workers never touch a mutex, and a waker of spinning workers makes
// one load of `sleeping` and no system call.
//
// A post only means "re-check": a waker may clear the bit of a later sleep
// than the one it meant to end. The worker then re-checks, finds nothing new,
// and sleeps again.
// -----------------------------------------------------------------------------

// Post to the sleepers among `workers` whose sleeping bit this call clears.
// Return the number of posts.
int Pool::wakeSleepers(SlotMask workers) {
  if (!workers)
    return 0;
  SlotMask m = sleeping_.fetchClear(workers) & workers;
  for (int slot : m)
    threads[slot].post();
  return m.count();
}

// Wake up to n sleeping workers that a release claimed. Used by each leaving
// worker, so that sleepers are woken as a tree, and not by the leader.
void Pool::wakeReleased(int n) {
  while (n > 0) {
    SlotMask s = sleeping_.load();
    s = s & released_.load();
    if (!s)
      return;
    n -= wakeSleepers(s.lowest(n));
  }
}

// Wait for work or for release. Return true for work, with `last` updated to
// the new epoch, and false when a release has claimed this worker.
//
// While it spins, it yields only with POMP_YIELD: once every that many spins.
bool Pool::waitWork(Thread &t, SlotMask self, uint64_t &last) {
  long every = settings.yieldEvery();
  SpinYield spinner(every == kYieldUnset ? 0 : every);
  for (long i = 0; i < settings.spin(); ++i) {
    uint64_t g = t.epoch(Order::Acquire);
    if (g != last) {
      last = g;
      return true;
    }
    if (released_.load(Order::Relaxed) & self)
      return false;
    spinner.spin();
  }
  uint64_t g;
  for (;;) {
    sleeping_.fetchSet(self);
    g = t.epoch();
    bool rel = bool(released_.load() & self);
    if (g != last || rel) {
      // A waker that cleared the bit has posted, or will: consume it.
      if (!(sleeping_.fetchClear(self) & self))
        t.waitPost();
      break;
    }
    t.waitPost();
    g = t.epoch();
    rel = bool(released_.load() & self);
    if (g != last || rel)
      break;
  }
  if (g == last)
    return false;
  // Work and release never come together: both claim the busy bit.
  last = g;
  return true;
}

// -----------------------------------------------------------------------------
// Recruitment
// -----------------------------------------------------------------------------

SlotMask Pool::idle() const {
  return settings.limitSlots() - busy_.load(Order::Relaxed);
}

SlotMask Pool::recruit(Team &team, int want, SlotMask prefer) {
  SlotMask old = busy_.load(Order::Relaxed);
  for (;;) {
    SlotMask idle = settings.limitSlots() - old;
    // Same request, same workers all idle: O(1).
    SlotMask pick = team.cachedPick(want, idle);
    if (!pick) {
      // Take what is idle now, preferred workers first; fewer is fine.
      pick = (idle & prefer).lowest(want);
      pick |= (idle - pick).lowest(want - pick.count());
    }
    if (!pick)
      return pick;
    if (busy_.compareExchangeWeak(
            old, old | pick, Order::Acquire, Order::Relaxed)) {
      if (pick.count() == want)
        team.remember(want, pick);
      return pick;
    }
  }
}

// Return the workers of a team. A release that ran meanwhile, from another
// host thread, did not see them idle; they are released here instead. Both
// sides are seq_cst, so one of them sees the other.
void Pool::returnWorkers(SlotMask workers) {
  busy_.fetchClear(workers);
  if (isClosed())
    releaseIdle();
}

// -----------------------------------------------------------------------------
// Release helpers
// -----------------------------------------------------------------------------

// The thread is no longer leaving the pool.
class NotLeaving {
public:
  explicit NotLeaving(const Thread &t) : thread_(t) {}
  bool operator()() const {
    return thread_.presence(Order::Acquire) != Thread::kLeaving;
  }

private:
  const Thread &thread_;
};

// Spin on a condition outside of any team (reclaim), yielding when long.
template <class Cond>
void spinUntil(Cond cond) {
  for (long n = 0; !cond();) {
    if (++n < kYieldSpin)
      relax();
    else
      sched_yield();
  }
}

// -----------------------------------------------------------------------------
// The pool
// -----------------------------------------------------------------------------

int Pool::enter(int slot) {
  int limit = settings.limitAcquire();
  if (!limit)
    return POMP_ERR_CLOSED; // Closed until an open gives the limit.
  if (slot < 1 || slot >= limit)
    return POMP_ERR_SLOT;
  // A thread in the pool cannot enter it again, and the leader cannot enter
  // it from inside its regions.
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  Thread &self = threads[slot];
  SlotMask me = SlotMask::of(slot);
  // Take the slot. If its previous thread was released but has not left yet,
  // wait for it: this costs this thread, not the leader. A slot whose thread
  // has not been released is in use.
  for (;;) {
    int observed;
    if (self.claimSlot(observed))
      break;
    if (observed == Thread::kInPool && !(released_.load() & me))
      return POMP_ERR_SLOT;
    sched_yield();
  }

  // Ready this slot's sleep before the slot can be recruited, posted to or
  // claimed by a release: all of these follow the clear of `busy` below.
  self.initPost();
  SelfWord::set(SelfWord(slot));
  self.regions().startAt(0, 0);
  self.clearPost();
  uint64_t epoch = self.epoch();

  // Become idle, hence recruitable, then check the gate. Release closes the
  // gate before it claims idle workers, so either this thread sees it closed
  // or the release claims it.
  busy_.fetchClear(me);
  if (isClosed()) {
    if (!(busy_.fetchSet(me) & me)) {
      // Nobody claimed this slot: back out.
      SelfWord::set(SelfWord());
      self.setPresence(Thread::kFree, Order::Release);
      return POMP_ERR_CLOSED;
    }
    // Claimed between the two steps, by a release or by a fork after an
    // open: serve as any worker.
  }
  numWorkers_.fetchAdd(1);

  while (waitWork(self, me, epoch))
    self.assignedTeam().runMember(self, slot, self.assignedTid());

  // Released. The busy bit stays set until a thread enters as this slot.
  self.regions().startAt(0, 0);
  self.setPresence(Thread::kLeaving);
  released_.fetchClear(me);
  wakeReleased(2);
  SelfWord::set(SelfWord());
  numWorkers_.fetchSub(1);
  self.setPresence(Thread::kFree, Order::Release);
  return POMP_OK;
}

// Claim every idle worker for a release, as a fork would, and start their
// wake-up. Called by a release after it closed the gate, and by a leader that
// returns its team's workers while the gate is closed.
void Pool::releaseIdle() {
  SlotMask old = busy_.load(), idle;
  do {
    idle = settings.limitSlots() - old - SlotMask::of(0);
  } while (idle && !busy_.compareExchangeWeak(old, old | idle));
  if (!idle)
    return;
  released_.fetchSet(idle);
  // Spinning workers see their bit and leave; each leaving worker wakes
  // sleeping ones. Only if every claimed worker sleeps is there nobody to start
  // that chain: then post to one.
  if ((sleeping_.load() & idle) == idle)
    wakeSleepers(SlotMask::of(idle.lowest()));
}

int Pool::close() {
  // Not from a worker, nor from inside the regions of the calling thread.
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  // Close the gate first, so an entry from now on either backs out or is
  // claimed just below.
  closeGate();
  releaseIdle();
  return POMP_OK;
}

int Pool::waitEmpty() {
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  int limit = settings.limitAcquire();
  // Wake any claimed worker still asleep, until all have left.
  while (released_.load() & settings.limitSlots()) {
    wakeReleased(kMaxThreads);
    sched_yield();
  }
  for (int i = 1; i < limit; ++i)
    spinUntil(NotLeaving(threads[i]));
  return POMP_OK;
}

int Pool::open(int threadLimit) {
  if (SelfWord::current().inPomp())
    return POMP_ERR_STATE;
  // Whether the pool is open or closed, the outcome is the same: open with
  // the process's limit, or POMP_ERR_LIMIT with the pool unchanged.
  int rc = settings.setLimit(threadLimit);
  if (rc != POMP_OK)
    return rc;
  closed_.store(0);
  return POMP_OK;
}
