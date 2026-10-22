# pomp: a minimal pool-based OpenMP runtime

pomp replaces LLVM's libomp as `libompruntime.a` when onnx-mlir is configured with
`-DONNX_MLIR_USE_POMP=ON`. It implements only the OpenMP entry points that onnx-mlir models and
zDNNx actually call. Its worker threads are **donated by the host** (for example Java) and
**released when the host chooses**, typically after each inference. libomp cannot do this: its
workers stay until the process ends (see `astudy/analysis/openmp-calls-in-om.md`).

It targets z/OS, where the host owns the threads, but it is plain C11 plus pthreads and is
developed and tested on Linux and macOS.

## Supported subset

| Entry point | Notes |
|---|---|
| `__kmpc_fork_call` | Up to 16 shared arguments; nested calls supported (see below) |
| `__kmpc_global_thread_num` | Returns the thread's slot. Called by OpenMPIRBuilder code on every worker |
| `__kmpc_push_num_threads` | Applies to the next fork; clamped to the limit |
| `__kmpc_push_proc_bind` | Accepted and ignored |
| `__kmpc_serialized_parallel`, `__kmpc_end_serialized_parallel` | From `if(false)`; a team of 1 |
| `__kmpc_barrier` | Centralized sense-reversing barrier |
| `__kmpc_for_static_init_{4,4u,8,8u}`, `__kmpc_for_static_fini` | `schedule(static)` and `schedule(static, chunk)`. Results identical to libomp's "balanced" default |
| `omp_get_thread_num`, `omp_get_num_threads`, `omp_get_level`, `omp_get_active_level`, `omp_in_parallel`, `omp_get_max_threads`, `omp_get_thread_limit`, `omp_get_max_active_levels` | Queries only |

Not supported, and absent from the archive so that their use fails at link time: reductions
(`__kmpc_reduce*`), critical, atomic, single and master, tasks, dynamic and guided schedules,
ordered, threadprivate, locks, cancellation, affinity, OMPT and offload. A static schedule kind
other than the two above aborts at run time.

## Environment

| Variable | Meaning |
|---|---|
| `OMP_THREAD_LIMIT` | **Required**, in `[1, 64]`, initial thread included. Otherwise `pomp_init` returns `POMP_ERR_LIMIT`, and an implicit init from a fork aborts |
| `OMP_NUM_THREADS` | `n` or `n1,n2`: default team size at levels 1 and 2. Default is the limit |
| `OMP_MAX_ACTIVE_LEVELS` | 1 (default) or 2. As in libomp, a list in `OMP_NUM_THREADS` implies 2 |
| `POMP_SPIN` | Spin iterations an idle worker makes before sleeping (default 200000; 0 sleeps immediately) |

## Host interface (`pomp.h`)

```c
int pomp_init(void);              // validate OMP_THREAD_LIMIT; pool starts open
int pomp_donate_thread(int slot); // slot in [1, limit-1]; returns once released
int pomp_release_workers(void);   // close the pool, release donors; does NOT wait
int pomp_reclaim_workers(void);   // wait until released donors have returned (optional)
int pomp_open_workers(void);      // reopen the pool for the next cycle; does not wait
int pomp_is_open(void);
int pomp_num_donated(void);
int pomp_get_thread_limit(void);
```

| Code | Meaning |
|---|---|
| `POMP_OK` | Success. For `pomp_donate_thread`: the thread served, was released, and is back with the host |
| `POMP_ERR_LIMIT` | `OMP_THREAD_LIMIT` missing or not in `[1, 64]` |
| `POMP_ERR_SLOT` | Slot out of range, or another thread is donated as that slot and not released |
| `POMP_ERR_STATE` | Wrong caller or state: release/open from a worker or inside a region; a worker donating itself |
| `POMP_ERR_CLOSED` | Donation refused at once because the pool is closed (after a release, before the next open) |

### Design goal: the leader never waits for workers

The initial thread (the *leader*: the one that runs the model, slot 0) is on the critical path,
so nothing in the protocol makes it wait for workers to arrive or to leave:

- **Release does not wait for anything.** It closes the pool with one store, claims every idle
  worker with one CAS on `busy`, marks them in `released` with one atomic OR, and returns.
  Spinning workers notice their bit and leave by themselves. Each worker that leaves wakes up to
  two sleeping ones, so sleepers are woken as a tree. The leader makes a system call only when
  *every* claimed worker is asleep, so that nobody else could start that chain, and then for one
  post only. Measured on macOS, the leader spends 0.1–0.3 µs in release with spinning workers,
  and 1.5–2 µs when all of them sleep. The previous blocking release took 0.6–35 µs (spinning)
  and 10–45 µs (sleeping) at 4 to 16 threads.
- **Open does not wait for anything.** It is one store. Workers of the previous cycle that are
  still leaving keep their `busy` bit set until their slot is donated again, so a fork can never
  recruit them.
- **Forks never wait for donors.** A fork takes the workers that are idle at that moment, and a
  team may be smaller than requested. While the pool is closed, a fork runs on the leader alone.
- **Waiting is done by whoever needs it**, never by the leader. A donor whose slot is still
  being vacated by its previous thread waits for that thread itself. A host thread that wants
  all donors back calls `pomp_reclaim_workers`.

### Pool states

```
            pomp_open_workers()
   +----------------------------------+
   v                                  |
 OPEN ------ pomp_release_workers() --> CLOSED
   donations accepted                   donations return POMP_ERR_CLOSED at once
   forks recruit idle donors            forks run on the leader alone
                                        released donors return to the host
```

The pool starts OPEN after `pomp_init`. A release while CLOSED releases any donor claimed since
the last release (normally none). An open while OPEN does nothing.

### Supported scenarios

Each scenario has a test in `test/pomp_host_test.c`. All of them pass at thread limits 2, 3, 4,
8, 16 and 17, with `POMP_SPIN` set to 0, 30 and 200000, and run clean under ThreadSanitizer.

**S1. Fresh threads per inference, killed after use** (`fresh`).

```
open; start N-1 threads, each pomp_donate_thread(i); wait until pomp_num_donated() == N-1;
run the model; release; join (or let exit) the N-1 threads
```

Once release has returned, every donor is guaranteed to return from `pomp_donate_thread`
without anything else being called. Joining them is therefore safe at once, and it costs the
joining thread, not pomp. A donor may exit as soon as it returns: pomp keeps no state on the
thread and registers no thread-exit destructor.

**S2. A host thread pool, reused across inferences** (`reuse`).

```
per inference: open; each pool thread calls pomp_donate_thread(i) again; run; release
```

A thread that returned from `pomp_donate_thread` is indistinguishable from one that never
called it. It may donate again as the same slot or another one, or act as the leader (S5).
`pomp_reclaim_workers()` lets the host wait for its threads, from any thread that is not a
donated worker. It is never required for correctness.

**S3. Server mode: many inferences per donation** (`server`).

```
open; donate N-1 threads once; run M inferences; release once
```

Between inferences the workers stay donated: they spin for `POMP_SPIN` iterations and then
sleep, with no cost to the leader beyond the fork. Releasing after every K inferences is S2.

**S4. Donations racing with release, and late donations** (`late`).

A thread may enter `pomp_donate_thread` at any time relative to a release:

- **Before** the release reads `busy`: the release claims it, and it returns `POMP_OK`.
- **During** the release, between its own two steps: release closes the gate *before* claiming.
  A donor clears its busy bit *before* reading the gate. Either the donor sees the gate closed,
  takes its bit back and returns `POMP_ERR_CLOSED` (counted nowhere), or the release sees the
  bit and claims it. Both orders were exercised: the test sees both outcomes.
- **After** the release, before the next open: `POMP_ERR_CLOSED` at once.

In every case the donor returns to the host without anyone waiting for it, and
`pomp_num_donated()` is 0 once those returns have completed. No donation can stay stuck in pomp
across a release. This was the deadlock risk of the previous version.

**S5. Threads changing slots and roles** (`swap`).

A host thread may donate as slot 3 in one cycle and as slot 5 in the next, and a thread that
donated before may later be the leader. The only rule is one thread per slot at a time.

**S6. The next cycle starting while the last one is still leaving** (`overlap`).

```
open; donate; run; release; open; donate; run; ...   (no reclaim in between)
```

Workers of the previous cycle that have not left yet are never recruited, because their
`busy` bit stays set. A host thread donating again to a slot whose previous donor is still
leaving waits, in its own thread, for that donor to finish. Regions in the new cycle use
whoever has arrived.

**S7. Release while every worker sleeps** (`sleepers`, run with `POMP_SPIN=0`).

The leader posts to one sleeper, which wakes two more as it leaves, and so on, so every donor
returns. The leader pays one post.

**S8. Misuse** (`errors`).

- Slot 0 or a slot `>= limit`: `POMP_ERR_SLOT`.
- A slot held by a donor that has not been released: `POMP_ERR_SLOT`.
- Release or open from inside a parallel region, or from a donated worker: `POMP_ERR_STATE`.

### Rules for the host

1. Call release and open from the leader, outside parallel regions. They are cheap, but they
   are not synchronized against a fork in progress on another thread.
2. One thread per slot at a time. Using fixed slot numbers per host thread is simplest.
3. **Never kill a thread while it is inside `pomp_donate_thread`.** It may be a member of a team
   at that moment, and the leader would wait for it forever. A thread is outside once
   `pomp_donate_thread` has returned. This is the one way to deadlock pomp from outside, and
   pomp cannot detect it.
4. To have exactly N-1 workers in the first region, wait until `pomp_num_donated()` reaches
   N-1 after donating. Without that, the first regions run with whoever has arrived, which is
   correct but slower.
5. Only one leader at a time: parallel regions started at the same time by two threads that are
   not donated workers are a fatal error.

## Example: the calls of one parallel loop, and what each costs

The loop below is written the way onnx-mlir models have it: `scf.parallel` lowers to
`omp.parallel` around `omp.wsloop`. The call sequence was produced by
`clang -O1 -fopenmp -fopenmp-enable-irbuilder`, the same OpenMPIRBuilder code generator that MLIR
uses.

```c
void scale(float *a, long n, float s) {
#pragma omp parallel
  {
#pragma omp for
    for (long i = 0; i < n; ++i)
      a[i] *= s;
  }
}
```

```
scale(a, n, s):                                          // on the leader
  __kmpc_fork_call(&loc, 1, scale..omp_par, &{n, a, s})

scale..omp_par(gtid*, tid*, args):                       // on each of the T team members
  g  = __kmpc_global_thread_num(&loc)
  __kmpc_for_static_init_8u(&loc, g, 34 /*static*/, &last, &lb, &ub, &stride, 1, 0)
  for (i = lb; i <= ub; ++i) a[i] *= s;                  // this thread's block of [0, n)
  __kmpc_for_static_fini(&loc, g)
  g2 = __kmpc_global_thread_num(&loc)
  __kmpc_barrier(&loc, g2)                               // end of omp for (no nowait)
```

When `num_threads(k)` is given, the leader first calls `__kmpc_push_num_threads(&loc, g, k)`,
which is one store. A combined `#pragma omp parallel for`, which zDNNx uses, emits the same
calls minus the barrier, because the join already synchronizes.

Cost of each step, where *T* is the number of threads in the region (leader included) and *W*
is the number of those workers that are asleep rather than spinning:

| Step | Who | Cost | What it does |
|---|---|---|---|
| `__kmpc_fork_call`: recruit | leader | **O(1)** when the same team as last time is free (one CAS on `busy`); O(T) bit scan otherwise | Cached mask hit in the steady state |
| `__kmpc_fork_call`: describe | leader | O(T) | Writes the tid→slot map and copies `argc` pointers (plus an O(T) split of spare workers, only when 2 levels are enabled) |
| `__kmpc_fork_call`: wake | leader | **O(T)** atomic increments (one `go` epoch per worker) + 1 load of `sleeping`; + O(W) posts (system calls) | No system call when every worker spins |
| `__kmpc_global_thread_num` | each thread | O(1) | One `__thread` load (or `pthread_getspecific`) |
| `__kmpc_for_static_init_8u` | each thread | O(1) | A division and a few multiplies; no shared memory touched |
| loop body | each thread | O(n/T) | The work |
| `__kmpc_for_static_fini` | each thread | O(1) | Empty |
| `__kmpc_barrier` | each thread | O(1) per thread, O(T) contention on one counter | The last arriving thread bumps the epoch; the others spin on it |
| `__kmpc_fork_call`: join | leader | O(1) per arrival, O(T) contention on `arrived`; then one atomic AND on `busy` | Leader spins until `arrived == T-1` |

So, per region:

- **Each worker does O(1) runtime work** besides its share of the loop: about five calls and no
  system call.
- **The leader does O(T) work**: it bumps one epoch per worker, and it waits on a counter
  that T−1 threads increment. Everything else on the leader's side is O(1), including
  recruitment in the steady state.
- **The barrier and the join are O(T) in wall time**, since T−1 atomic increments of one cache
  line are serialized. At T ≤ 64 this is about 1–4 µs at 16 threads on the test machine (see
  the latency table below).

Both O(T) parts on the leader could be made O(log T): the wake-up by having woken workers wake
others (as release already does), and the barrier and join by a combining tree. Neither is done
yet.

### Cost compared with libomp (macOS)

Measured on an 18-core Apple Silicon laptop (macOS, arm64), with Homebrew libomp 22.1.4 and
LLVM clang. Times are µs per region, with the median of 3 runs of 100000 regions; 32 threads
is a single run of 10000.

- `fork/join`: an empty `#pragma omp parallel` (`pomp_test bench`).
- `barrier`: one `#pragma omp barrier` inside a running region.
- `loop`: the `scale` example above on 1024 floats (`pomp_test loop`): fork, static init, a
  64-element block per thread at 16 threads, barrier, join.

libomp is shown twice:

- **default**: on this hybrid CPU, libomp's default blocktime is 0. Idle workers go to sleep at
  once, so every region pays an OS wake-up for each worker.
- **`KMP_BLOCKTIME=200`**: workers spin for 200 ms before sleeping, which is libomp's default
  on non-hybrid CPUs such as z. This is the fair comparison with pomp, which spins
  `POMP_SPIN` iterations before sleeping.

| Threads | libomp default<br>fork/join · barrier · loop | libomp `KMP_BLOCKTIME=200`<br>fork/join · barrier · loop | pomp<br>fork/join · barrier · loop |
|---|---|---|---|
| 1 | 0.04 · 0.00 · 0.10 | 0.04 · 0.00 · 0.10 | **0.00 · 0.00 · 0.04** |
| 4 | 10.8 · 10.6 · 22.2 | 0.40 · 0.27 · 0.76 | **0.40 · 0.18 · 0.62** |
| 8 | 25.2 · 25.6 · 50.3 | 1.23 · 0.78 · 2.26 | **1.41 · 0.49 · 2.11** |
| 16 | 49.4 · 48.7 · 98.4 | **1.68 · 1.25 · 3.22** | 3.42 · 0.75 · 4.65 |
| 32 (oversubscribed) | 114 · 115 · 235 | did not finish in 100 s | 150 · 45 · 302 |

Reading the table against the cost model above:

- **1 thread**: pomp runs the region inline, with no team, no wake-up and no barrier, so it
  costs a function call. libomp still goes through its fork path.
- **4–8 threads**: pomp matches or beats spinning libomp. Its barrier is about 35% faster, and
  the whole loop region is about 20% faster at 4 threads and 7% faster at 8. Its fork/join is
  equal at 4 threads and about 15% slower at 8.
- **16 threads**: pomp's fork/join is about 2× spinning libomp's (3.4 against 1.7 µs). This is
  the O(T) part on the leader: it bumps 15 epochs one after another, then waits on one counter
  that 15 threads increment. libomp wakes and gathers workers through a tree (its default
  "hyper" barrier), which is O(log T). pomp's barrier is still faster, and the whole loop region is
  about 45% slower. The tree wake-up and join noted above are what would close this gap.
- **32 threads on 18 cores**: both runtimes collapse, as expected when spinning threads outnumber
  cores. Spinning libomp did not complete. These numbers say nothing about z/OS, where the thread
  limit should not exceed the processors available to the job.
- **libomp's default on this machine** is 20–50× slower than either spinning runtime, because
  every region sleeps and wakes its workers. That is a property of the hybrid-CPU default
  (blocktime 0), not of libomp's algorithms.

## Design

**Thread identity.** Each thread's slot is kept in one `__thread int32_t`, set when the thread
is donated. A thread that was never donated reads 0. Build with `-DPOMP_USE_PTHREAD_KEY` (CMake:
`-DPOMP_USE_PTHREAD_KEY=ON`) to use a pthread key instead, if `__thread` is unavailable in a
static archive linked into `model.so`. Only `__kmpc_global_thread_num` and the `omp_*` queries
read it; every other entry point receives the slot as `gtid`.

**One word for the pool.** `pomp_busy` holds one bit per slot. A bit is clear iff that slot is a
donated, idle worker. All slots start busy:
- A donor clears its own bit when it arrives. No team includes it yet, so there is no race.
- A leader recruits its whole team with one CAS (`busy |= pick`).
- After the join, the leader returns the team with one atomic AND.
- `pomp_release_workers` claims every idle bit the same way. Released slots stay busy until
  they are donated again.

**Static descriptors.** Nothing is allocated after start-up:
- `pomp_threads[64]`: per-slot wake-up epoch and state, each on its own cache line; the
  current assignment; the region stack; and the sleep mutex and condition variable.
- `pomp_root_team`: the team the root leads at level 1.
- `pomp_slot_team[64]`: the team that slot *i* leads at level 2.

A team descriptor holds the outlined function and arguments, the team size, the tid-to-slot
map, the join counter, the barrier, and the reuse cache.

**Fork.**
1. Decide the team size: the pushed `num_threads`, otherwise the `OMP_NUM_THREADS` entry for
   the current level, clamped to the limit.
2. If the team size is 1, or the thread is already at `OMP_MAX_ACTIVE_LEVELS`, run inline as
   a team of 1. This matches libomp's default for the zDNNx matmul nest.
3. Otherwise recruit. If the request matches the team's cached one and those same workers are
   all idle, the pick is O(1). Otherwise take the lowest idle bits, preferring the team's
   preferred set.
4. Write the descriptor, set each member's `team` and `tid`, and bump its `go` epoch.
5. Run as tid 0, wait until `arrived == nth - 1`, then clear the team's bits.

**Two levels.** When a level-1 team forks with `OMP_MAX_ACTIVE_LEVELS=2`, the leader splits the
remaining idle workers into one contiguous group per member. Each member's level-2 fork prefers
its own group. Inner teams therefore do not compete with each other, and their caches stay
valid from one region to the next. For example, with 16 threads, a 4×4 split gives the same
4 + 4×3 assignment every time.

**Wake-up without losing a signal.** An idle worker spins for `POMP_SPIN` iterations on two
things: its own `go` epoch (work) and its bit in the global `released` word (release). The
`released` word is written only by release and by leaving workers, so it stays shared in every
cache and costs the spin nothing. To sleep, the worker sets its bit in the global `sleeping`
word, re-reads `go` and `released`, and waits on its condition variable.

A waker (a fork, a release, or a leaving worker) first publishes: it bumps `go`, or sets the
`released` bit. It then reads `sleeping`, clears with one atomic AND the bits of the workers it
is waking, and posts only to those whose bit it cleared. All of these accesses are sequentially
consistent, so either the worker sees the new value or the waker sees the bit. No wake-up is
lost, and exactly one side clears each bit. A waker of spinning workers makes one load and no
system call, and a spinning worker never touches its mutex.

The worker treats a post only as "re-check", never as "run". A post can be stale: the waker may
clear the bit of a later sleep than the one it meant to end. An earlier version that treated a
post as "run" hung under the 8-thread fork benchmark; the re-check loop fixes it.

**Release.** It closes the gate, claims the idle workers in `busy`, ORs them into `released`,
and returns; it posts once only if all of them sleep. A released worker sets its slot to
LEAVING, clears its `released` bit, wakes up to two released sleepers, resets its thread slot to
0, decrements the donation count, and sets its slot to FREE. Its `busy` bit stays set until the
slot is donated again, so no fork can recruit a leaving worker.

**Donation and the gate.** A donor takes its slot (FREE to DONATED; if the slot is LEAVING it
yields until FREE), resets its descriptor, and clears its `busy` bit. Only then does it read
`closed`. Release writes `closed` before it reads `busy`. With sequentially consistent accesses,
either the donor sees the gate closed or the release sees the cleared bit. If the donor sees it
closed, it sets its bit back with an atomic OR. If the bit was still clear, nobody claimed the
slot, and the donor backs out with `POMP_ERR_CLOSED`. Otherwise a release (or a fork after a
new open) claimed it in between, and it serves as an ordinary worker.

**Barrier.** Each arriving thread increments the team's counter. The last one resets the
counter and bumps the barrier epoch; the others spin, then yield, until the epoch changes. The
compiler emits this barrier after every worksharing loop.

**Join.** Each member's last access to the team is its increment of `arrived`, with release
ordering. The leader then owns the descriptor again.

## Testing

Tests are in `test/`. Each test program builds against both pomp (`-DPOMP`, with donor threads
standing in for the host) and libomp, so their outputs can be compared.

| Test | Checks | Result (macOS arm64, 18 cores) |
|---|---|---|
| `pomp_test static` | Owner of every iteration for 5 IV types × 13 trip counts × 4 steps × chunked, for every team size up to the limit | Digest identical to libomp at limits 1, 2, 3, 4, 5, 8, 16 |
| `pomp_test nested` | Team size at each level of the zDNNx 3-level `if` nest; `collapse(4)` | Identical to libomp with nesting off, `OMP_NUM_THREADS=4,4`, and `OMP_MAX_ACTIVE_LEVELS=2` (including partial 4/2 teams at limit 8) |
| `pomp_test stress N` | Many regions of varying size with a barrier | 200000 regions, no errors, at `POMP_SPIN` 200000, 50 and 0 |
| `pomp_test cycles N` | Donate, run 10 regions, release: the per-inference pattern | 20000 cycles, no errors; about 200 µs per cycle at limit 8, mostly `pthread_create`/`join` in the test host |
| `pomp_host_test <scenario>` | Host protocol scenarios S1–S8 above, and `latency` (leader time in release) | All pass at limits 2–17, `POMP_SPIN` 0, 30 and 200000, with 1500-cycle soak runs; no ThreadSanitizer reports |
| `pomp_irbuilder_test` | Code from `-fopenmp-enable-irbuilder`, the OpenMPIRBuilder path MLIR uses, which calls `__kmpc_global_thread_num` on workers | Correct at limits 1, 4, 18; also with `POMP_USE_PTHREAD_KEY` |
| ThreadSanitizer | `static`, `nested` (both nesting modes), `stress`, `cycles`, and all of `pomp_host_test` | No reports |

Measured costs against libomp are in "Cost compared with libomp" in the example section above.

## Building outside onnx-mlir

```sh
clang -O2 -std=c11 -c pomp.c
clang -O2 -fopenmp -DPOMP -c test/pomp_test.c && clang pomp_test.o pomp.o -o pomp_test
OMP_THREAD_LIMIT=8 ./pomp_test static
```

## Open items

- **Not yet built or run on z/OS.** Still to confirm there: `__thread` in a static archive
  linked into a model `.so`, and `atomic_*` codegen. The cache line is set to 256 bytes for
  s390x.
- **No end-to-end test with an onnx-mlir model yet.** The installed `onnx-mlir` binary currently
  fails to load because of a Homebrew abseil upgrade.
- **zDNNx still compiles to `_4u`, `_8` and the serialized pair.** pomp supports them, so the
  source simplification in `openmp-calls-in-om.md` §4.2 is optional.
