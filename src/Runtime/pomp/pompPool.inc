/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------------- pompPool.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: the pool of threads. Pool, the allocation of the pool's
// descriptors, the idle wait and its wake-ups, recruitment, and the host
// protocol: open, entry, close, release.
//
//===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------------
// Terms used in Pool.
//
// Idle: a pool thread in no team, waiting for work; its busy bit is clear.
//
// Recruit: a fork claims idle pool threads as its workers, setting their busy
//   bits. After the join, they are returned: idle again.
//
// Release: the pool close claims every idle pool thread, as a fork would, but
//   sets its released bit instead of giving it work. Threads in a team are
//   released when returned.
//
// Leave: a released thread returns from pomp_enter_pool to the host.
// -----------------------------------------------------------------------------

class Pool {
public:
  // At init: the lock of open and close.
  void init();
  // Whether the pool's descriptors exist; acquired before reading them.
  bool allocatedAcquire() const { return allocated_.load(Order::Acquire); }
  // The host calls of pomp.h, in the order of a cycle: pomp_open_pool,
  // pomp_enter_pool, pomp_close_pool and pomp_wait_pool_empty.
  int open(int fallback, int *limit);
  int enter(int32_t slot);
  int close();
  int waitEmpty();
  bool isOpen(Order o = Order::SeqCst) const { return state_.load(o) == kOpen; }
  int size() const { return numPoolThreads_.load(); }
  // Fork that finds the pool not open: warn once if it was never opened.
  void warnIfNeverOpened();

  // Recruit up to `want` idle pool threads as workers, preferring `prefer`;
  // return them after the join.
  SlotMask recruit(Team &team, int want, SlotMask prefer);
  // The idle pool threads (relaxed).
  SlotMask idle() const;
  // Post to the sleepers among `workers`; see "Wake-up and sleep".
  void wakeSleeping(SlotMask workers);
  // After the join: make a team's workers idle again.
  void returnWorkers(SlotMask workers);

private:
  // The first open: allocate the descriptors of `n` slots; false if out of
  // memory.
  bool allocate(int n);
  // Idle pool thread: wait for work (true) or for release (false).
  bool waitWork(Thread &t, SlotMask self, uint64_t &last);
  // Leaving pool thread: wake up to `n` released sleepers.
  void wakeReleased(int n);
  // Claim and release every idle pool thread.
  void releaseIdle();
  // Post to the sleepers among `workers` whose bit this call clears.
  int wakeSleepers(SlotMask workers);

  // Slot i is out iff it is an idle pool thread. Every slot starts busy.
  alignas(kCacheLine) AtomicMask busy_{SlotMask::below(kMaxThreads)};
  // Slot i is in while pool thread i sleeps, or is about to.
  alignas(kCacheLine) AtomicMask sleeping_;
  // Slot i is in from the release that claimed pool thread i until it leaves.
  alignas(kCacheLine) AtomicMask released_;
  // kNeverOpened (zero) at start, then kOpen and kClosed by open and close.
  static constexpr int kNeverOpened = 0;
  static constexpr int kOpen = 1;
  static constexpr int kClosed = 2;
  alignas(kCacheLine) Atomic<int> state_;
  // Serializes open and close: an open that finds the pool open fails.
  Mutex openMutex_;
  Atomic<int> allocated_; // The descriptors exist.
  Atomic<int> numPoolThreads_;
  Atomic<int> neverOpenedSeen_;
};

POMP_CONSTINIT Pool pool;

// -----------------------------------------------------------------------------
// Wake-up and sleep
//
// An idle pool thread spins on its `go` epoch and its `released` bit, then
// sleeps with its bit set in `sleeping`. A waker publishes, then posts only to
// the sleepers whose bit it clears. Waking spinning threads costs one load of
// `sleeping`, and no system call.
// -----------------------------------------------------------------------------

inline void Pool::wakeSleeping(SlotMask workers) {
#ifdef POMP_STATS
  stat(kStatRegions);
  stat(kStatForkPosts, uint64_t(wakeSleepers(sleeping_.load() & workers)));
#else
  wakeSleepers(sleeping_.load() & workers);
#endif
}

// Post to the sleepers among `workers` whose sleeping bit this call clears.
// Return the number of posts.
int Pool::wakeSleepers(SlotMask workers) {
  if (!workers)
    return 0;
  SlotMask m = sleeping_.clearAndReport(workers);
  for (int32_t slot : m)
    threads[slot].post();
  return m.count();
}

// Wake up to n sleeping released threads. Each leaving thread calls it, so
// the wake-ups spread as a tree, not from the leader.
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
// the new epoch, and false when a release has claimed this thread.
bool Pool::waitWork(Thread &t, SlotMask self, uint64_t &last) {
  bool adapt = settings.spinAdaptive();
  long maxUs = settings.spinMaxUs();
  uint64_t t0 = adapt ? nowUs() : 0;
  for (SpinBudget spin(adapt ? t.idleSpin_.next(maxUs) : settings.spinUs());
       spin.more();) {
    uint64_t g = t.epoch(Order::Acquire);
    if (g != last) {
      // Thread t got some new work.
      last = g;
      if (adapt)
        t.idleSpin_.update(nowUs() - t0, maxUs);
      return true;
    }
    if (released_.load(Order::Relaxed) & self) {
      // Thread t was just released and will thus leave the pool.
      return false;
    }
    relax();
  }
  // Sleep: set our bit, then re-check. A waker publishes, then reads
  // `sleeping`; all seq_cst, so either we see its publish or it sees our
  // bit: no wake-up is lost.
  uint64_t g;
  for (;;) {
    POMP_STAT(kStatIdleSleeps);
    sleeping_.fetchSet(self);
    g = t.epoch();
    bool rel = bool(released_.load() & self);
    if (g != last || rel) {
      // Leave the sleepers. If a waker cleared our bit first, it has posted
      // or will: wait for that post, so that it does not stay pending and
      // end a later sleep of this thread.
      if (!sleeping_.clearAndReport(self))
        t.waitPost();
      break;
    }
    // A post only means "re-check": it may be meant for an earlier sleep.
    t.waitPost();
    g = t.epoch();
    rel = bool(released_.load() & self);
    if (g != last || rel)
      break;
  }
  // Work and release never come together: both claim the busy bit.
  if (g == last) {
    // Exited the loop without getting new work (updated epoch), Thread t was
    // released.
    return false;
  }
  // Exited the loop with new work (updated epoch), Thread t got some new work.
  last = g;
  if (adapt)
    t.idleSpin_.update(nowUs() - t0, maxUs);
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
      // Take what is idle now, preferred threads first; fewer is fine.
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

// Return a team's workers. A release that ran meanwhile did not see them
// idle, so release them here; both sides are seq_cst, so one sees the other.
void Pool::returnWorkers(SlotMask workers) {
  busy_.fetchClear(workers);
  if (!isOpen())
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

// Spin on a condition outside of any team, yielding when long.
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

// The pool starts closed: a host that sets the limit but never opens the pool
// runs every region on one thread, and is told why. Not after a close, where
// one thread is the documented behaviour.
void Pool::warnIfNeverOpened() {
  if (state_.load(Order::Relaxed) == kNeverOpened &&
      !neverOpenedSeen_.exchange(1))
    fprintf(stderr, "pomp: warning: a parallel region runs on one thread, as "
                    "the pool was never opened; call pomp_open_pool() before "
                    "the pool threads enter (see pomp.h)\n");
}

void Pool::init() { openMutex_.init(); }

bool Pool::allocate(int n) {
  // One block, never freed: `n` threads, then `n` teams from the next line,
  // for the pool threads; index 0 is unused (roots have their own).
  size_t tBytes =
      (size_t(n) * sizeof(Thread) + kCacheLine - 1) & ~size_t(kCacheLine - 1);
  char *p =
      static_cast<char *>(allocLines(1, tBytes + size_t(n) * sizeof(Team)));
  if (!p)
    return false;
  Thread *t = reinterpret_cast<Thread *>(p);
  Team *m = reinterpret_cast<Team *>(p + tBytes);
  // Not yet visible to other threads: open publishes them, with allocated_,
  // once the limit is set. Pool threads init their sleep on entry.
  threads = t;
  teams = m;
  for (int32_t i = 1; i < n; ++i) {
    t[i].initIdentity(i, nullptr, &m[i]);
    m[i].init(i, t[i]);
  }
  return true;
}

// One caller opens a cycle: the others find it open and get POMP_ERR_OPEN,
// with the limit. The first open allocates the descriptors for
// OMP_THREAD_LIMIT, else for `fallback`, and only then fixes the limit: on
// failure nothing is fixed, and a later open may retry, with another
// fallback. All before the state store that lets threads enter.
int Pool::open(int fallback, int *limit) {
  if (Thread::callerInPomp())
    return POMP_ERR_STATE;
  if (!limit || fallback < 0 || fallback > kMaxThreads)
    return POMP_ERR_LIMIT;
  ScopedLock<Mutex> lock(openMutex_);
  *limit = settings.limit();
  if (isOpen())
    return POMP_ERR_OPEN;
  if (!allocated_.load(Order::Relaxed)) {
    int lim = settings.limit() ? settings.limit() : fallback;
    if (!lim)
      return POMP_ERR_LIMIT;
    if (!allocate(lim))
      return POMP_ERR_NOMEM;
    if (!settings.limit())
      settings.setLimit(lim);
    // Publishes the descriptors, and the limit: an entry that sees them also
    // sees the limit, so its slot check is right (see allocatedAcquire).
    allocated_.store(1, Order::Release);
  }
  *limit = settings.limit();
  // Freeze before the store that lets forks recruit (see Settings::freeze).
  settings.freeze();
  state_.store(kOpen);
  return POMP_OK;
}

int Pool::enter(int32_t slot) {
  if (!allocatedAcquire())
    return POMP_ERR_CLOSED; // No descriptors before the first open.
  if (slot < 1 || slot >= settings.limit())
    return POMP_ERR_SLOT;
  // Not from a pool thread, nor from inside a region.
  if (Thread::callerInPomp())
    return POMP_ERR_STATE;
  Thread &self = threads[slot];
  SlotMask me = SlotMask::of(slot);
  // Take the slot; wait if its previous thread was released but has not left.
  for (;;) {
    int observed;
    if (self.claimSlot(observed))
      break;
    if (observed == Thread::kInPool && !(released_.load() & me))
      return POMP_ERR_SLOT;
    sched_yield();
  }

  // Init before the slot becomes recruitable, below. An initial thread (a
  // root) that enters keeps its Root for when it leaves.
  self.init();
  Thread *initial = Self::get();
  Self::set(&self);
  self.regions().startAt(0, 0);
  self.clearPost();
  uint64_t epoch = self.epoch();

  // Become idle, then check that the pool is open. A close marks it closed
  // before claiming the idle threads, so either we see it closed or the
  // close claims us.
  busy_.fetchClear(me);
  if (!isOpen()) {
    if (!(busy_.fetchSet(me) & me)) {
      // Nobody claimed this slot: back out.
      Self::set(initial);
      self.setPresence(Thread::kFree, Order::Release);
      return POMP_ERR_CLOSED;
    }
    // Claimed meanwhile, by a release or a fork: serve as any pool thread.
  }
  numPoolThreads_.fetchAdd(1);

  while (waitWork(self, me, epoch))
    self.assignedTeam().runWorker(self, slot, self.assignedTid());

  // Released. The busy bit stays set until a thread enters as this slot.
  self.regions().startAt(0, 0);
  self.setPresence(Thread::kLeaving);
  released_.fetchClear(me);
  wakeReleased(2);
  Self::set(initial);
  numPoolThreads_.fetchSub(1);
  self.setPresence(Thread::kFree, Order::Release);
  return POMP_OK;
}

// Claim every idle pool thread, as a fork would, mark it released, and start
// the wake-ups. Called once the pool is marked closed.
void Pool::releaseIdle() {
  SlotMask old = busy_.load(), idle;
  do {
    idle = settings.limitSlots() - old - SlotMask::of(0);
  } while (idle && !busy_.compareExchangeWeak(old, old | idle));
  if (!idle)
    return;
  released_.fetchSet(idle);
  // Spinning threads see their bit and leave, waking sleepers as they go. If
  // all claimed threads sleep, post to one to start that chain.
  if ((sleeping_.load() & idle) == idle)
    wakeSleepers(SlotMask::of(idle.lowest()));
}

int Pool::close() {
  // Not from a pool thread, nor from inside the regions of the calling thread.
  if (Thread::callerInPomp())
    return POMP_ERR_STATE;
  // Mark closed first: a concurrent entry then backs out or is claimed below.
  // Under the lock of open, so that opens and closes are ordered.
  ScopedLock<Mutex> lock(openMutex_);
  state_.store(kClosed);
  releaseIdle();
  return POMP_OK;
}

int Pool::waitEmpty() {
  if (Thread::callerInPomp())
    return POMP_ERR_STATE;
  if (!allocatedAcquire())
    return POMP_OK; // Never opened: no pool thread.
  int limit = settings.limit();
  // Wake any claimed thread still asleep, until all have left.
  while (released_.load() & settings.limitSlots()) {
    wakeReleased(kMaxThreads);
    sched_yield();
  }
  for (int32_t slot = 1; slot < limit; ++slot)
    spinUntil(NotLeaving(threads[slot]));
  return POMP_OK;
}
