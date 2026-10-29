/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- pompRoots.inc ----------------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Part of pomp.cpp: Root, the data of a thread that runs regions outside the
// pool (OpenMP's initial thread), the registry that hands Roots out, and the
// rules that make their reuse safe.
//
//===----------------------------------------------------------------------===//

// A root's data: its descriptor and the teams it leads, allocated together,
// recycled together and never freed; see "Roots" below for the rules that
// keep a late wake-up of a recycled Root harmless.
struct Root {
  Thread thread;
  Team levelOne;
  Team levelTwo;
  Root *next; // In the free list.
};

// The Roots: those in use, one per root, and the free list.
class RootRegistry {
public:
  // At init: the lock, and the key whose destructor recycles a Root.
  void init();
  // The calling thread, which has no descriptor, becomes a root.
  Thread &acquire();
  // The key's destructor: the thread of `r` has ended.
  void recycle(Root &r);

private:
  // A new Root, zeroed and initialized; under mutex_.
  Root *allocate();

  Mutex mutex_;
  Root *free_ = nullptr; // Under mutex_.
  int32_t count_ = 0;    // Roots allocated; under mutex_.
  pthread_key_t exitKey_{};
};

POMP_CONSTINIT RootRegistry rootRegistry;

// -----------------------------------------------------------------------------
// Roots
//
// A thread that is not in the pool becomes a root at its first region (or
// pushed team size): OpenMP's initial thread. It takes a Root, its descriptor
// and the two teams it leads, its own until it ends. So roots never share a
// descriptor, any number of them may run regions at once, and the next region
// of a root touches nothing global but the pool.
//
// When the thread ends, the destructor of `exitKey_` returns the Root to the
// free list, and a later root may take it. Roots are never freed.
//
// Why: a late wake-up. At the end of a region each worker stores its done
// flag, then wakes the leader (TeamJoin::workerFinish). Once the last flag is
// stored the leader may join, return, and its thread end; a worker may be
// delayed for any time between its store and its wake-up, so that wake-up
// may come while the Root is on the free list or owned by another root. It
// touches, in the Root:
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
//   1. Never free a Root, nor its Thread or teams: the late wake-up would use
//      freed memory, a pthread mutex among it.
//   2. Never initialize a Root's post again (Thread::init does it once): a
//      late waker may still be inside its post().
//   3. Never reset a team's unique ids on reuse: they must keep growing, or a
//      late or stale flag could match a new region's id. A Root keeps its
//      team indices too. RootRegistry::acquire resets only the pending team
//      size.
//   4. Recycle a Root whole: its teams' `leader_` must stay its Thread, as a
//      late wake-up of team T posts to T's leader_.
//   5. After its done store, a worker reads nothing of the team but fields
//      set at allocation and never changed (`leader_`, read before the
//      store, and the leader's slot): any other would belong to the region
//      of the Root's next owner.
// test/pomp_host_test.c latepost (S18) stresses this case.
// -----------------------------------------------------------------------------

// The destructor of RootRegistry::exitKey_, with the ending thread's Root.
void recycleRoot(void *r) {
  rootRegistry.recycle(*static_cast<Root *>(r));
}

void RootRegistry::init() {
  mutex_.init();
  checkPthread(
      pthread_key_create(&exitKey_, recycleRoot), "pthread_key_create");
}

Root *RootRegistry::allocate() {
  // Team indices: below kMaxThreads for the pool's teams, then two per Root.
  int64_t index = int64_t(kMaxThreads) + 2 * int64_t(count_);
  if (uint64_t(index + 1) >= kIdStride)
    fatal("too many host threads running OpenMP regions at once");
  Root *b = static_cast<Root *>(allocLines(1, sizeof(Root)));
  if (!b)
    fatal("cannot allocate the descriptor of a host thread");
  b->thread.init();
  b->thread.initIdentity(0, &b->levelOne, &b->levelTwo);
  b->levelOne.init(int32_t(index), b->thread);
  b->levelTwo.init(int32_t(index + 1), b->thread);
  ++count_;
  POMP_STAT(kStatRoots);
  return b;
}

// A reused Root keeps all of its state but the pending team size: see rules
// 2 and 3 of "Roots".
Thread &RootRegistry::acquire() {
  Root *b;
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

// The value of exitKey_ is already null here: use `r`, not Self. A thread
// that ends inside a region (pthread_exit) keeps its Root, never reused: its
// team may still use it.
void RootRegistry::recycle(Root &r) {
  Self::set(nullptr);
  if (r.thread.regions().depth() != 0)
    return;
  ScopedLock<Mutex> lock(mutex_);
  r.next = free_;
  free_ = &r;
}

inline Thread &Thread::currentOrNewRoot() {
  Thread *self = Self::get();
  return self ? *self : rootRegistry.acquire();
}
