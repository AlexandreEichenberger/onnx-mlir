# pomp: a minimal pool-based OpenMP runtime

pomp replaces LLVM's libomp as `libompruntime.a` when onnx-mlir is configured with
`-DONNX_MLIR_USE_POMP=ON`. It implements only the OpenMP entry points that onnx-mlir models and
zDNNx actually call. Its worker threads are **lent by the host** (for example Java) through a
**pool**, and **returned when the host closes the pool**, typically after each inference. libomp
cannot do this: its workers stay until the process ends (see
`astudy/analysis/openmp-calls-in-om.md`).

It targets z/OS, where the host owns the threads, and is developed and tested on Linux and
macOS. Its interface, `pomp.h`, is C. Its implementation, `pomp.cpp`, is C++17 that needs exactly
what a C library needs, the C library and pthreads; see "Language" below.

## Supported subset

| Entry point | Notes |
|---|---|
| `__kmpc_fork_call` | Up to 16 shared arguments, the same limit as standard libomp (see below); nested calls supported |
| `__kmpc_global_thread_num` | Returns the thread's slot. Called by OpenMPIRBuilder code on every worker |
| `__kmpc_push_num_threads` | Applies to the calling thread's next fork; clamped to the limit |
| `__kmpc_push_proc_bind` | Accepted and ignored |
| `__kmpc_serialized_parallel`, `__kmpc_end_serialized_parallel` | From `if(false)`; a team of 1. Like a fork, it makes a host thread a root (S9) |
| `__kmpc_barrier` | Central counter for small teams, 4-ary tree for large ones; long waits sleep |
| `__kmpc_for_static_init_{4,4u,8,8u}`, `__kmpc_for_static_fini` | `schedule(static)` and `schedule(static, chunk)`. Results identical to libomp's "balanced" default |
| `omp_get_thread_num`, `omp_get_num_threads`, `omp_get_level`, `omp_get_active_level`, `omp_in_parallel`, `omp_get_max_threads`, `omp_get_thread_limit`, `omp_get_max_active_levels` | Queries only. While no thread limit is set, `omp_get_max_threads` and `omp_get_thread_limit` return 1 |

**Argument limit.** `__kmpc_fork_call` passes the outlined function at most 16 shared arguments,
and aborts with a message above that. This is the same limit as standard libomp's portable
implementation, `kmp_invoke_microtask.cpp`, which libomp uses on z/OS: it handles 0 to 15
arguments and aborts above that, so pomp accepts one more. (libomp's assembly trampolines, used
on Linux for x86, AArch64 and s390x, have no fixed limit.) Neither code path in onnx-mlir comes
near it:

- MLIR's OpenMPIRBuilder packs every captured variable of an onnx-mlir model into one struct, so
  each fork passes exactly 1 argument.
- clang passes one pointer per shared variable for zDNNx, and its largest region passes 5.

**Nesting limit.** A thread can nest regions at most 256 levels deep, and aborts with a message
beyond that. Only the first 8 levels are recorded, and forks past them are serialized; the
deeper ones are only counted. Real programs nest two or three levels; only unbounded recursion
through parallel regions gets near the limit.

Not supported, and absent from the archive so that their use fails at link time: reductions
(`__kmpc_reduce*`), critical, atomic, single and master, tasks, dynamic and guided schedules,
ordered, threadprivate, locks, cancellation, affinity, OMPT and offload. A static schedule kind
other than the two above aborts at run time.

## Environment

Each setting below can also be set from code with `pomp_set_option(name, value)`, which takes the
variable's name and the same strings; for hosts that cannot set environment variables, such as a
JVM. A set environment variable takes precedence over calls (the call returns `POMP_OK` and
changes nothing), so an operator can still override at launch; `POMP_IGNORE_ENV=1` reverses that,
for hosts that must not be affected by variables meant for another OpenMP library in the process.
`POMP_SPIN` may change at any time and applies to the next waits; the others only before the
first `pomp_open_pool` (`POMP_ERR_STATE` after). The environment is read once, at pomp's first
call.

| Variable | Meaning |
|---|---|
| `OMP_THREAD_LIMIT` | In `[1, 64]`, initial thread included. Fixes the thread limit for the whole process, and takes precedence over `pomp_set_thread_limit` (see "The thread limit"). If unset, the first valid `pomp_set_thread_limit(limit)` gives it; until then the pool is closed, regions run on one thread, and `omp_get_max_threads` and `omp_get_thread_limit` return 1, with one warning on stderr. If set but invalid, it is ignored with a warning on stderr, as if unset. Nothing aborts |
| `OMP_NUM_THREADS` | `n` or `n1,n2`: default team size at levels 1 and 2, clamped to the limit once it is known. Default is the limit |
| `OMP_MAX_ACTIVE_LEVELS` | 1 (default) or 2; 0 means 1 and larger values mean 2. As in libomp, a list in `OMP_NUM_THREADS` implies 2 |
| `OMP_DISPLAY_ENV` | `TRUE` (or `1`) prints every setting on stderr once, at the first `pomp_open_pool`, in OpenMP's format, each followed by its source: default, environment, or the call that set it. `VERBOSE` adds pomp's build constants. Read from the environment only |
| `POMP_SPIN` | How long any wait spins before it sleeps: an idle worker, the leader at the join, a thread in a barrier. `auto` (default): each thread adapts, separately for waiting for work and for team waits, between 2 µs and 100 µs: after a wait of *d* µs, the next spin covers 2·*d* if *d* ≤ 100, and halves (down to 2) if *d* > 100. `auto:M` sets the ceiling to M µs (`auto:0` sleeps at once). A number N spins exactly N µs, without adapting (0 sleeps at once). Waits never yield. Why: on z a spinning thread slows an SMT sibling that computes by about 30%, so long waits should sleep; but a wake-up costs 2–10 µs, so short waits that sleep make the next ones longer, and the team keeps falling asleep (see `POMP_Z_Tuning.md`) |
| `POMP_TREE_BARRIER` | Smallest team that uses the tree barrier instead of the central one. Default 5 on s390x, never elsewhere; 0 or 2 forces the tree, 65 (`POMP_MAX_THREADS` + 1) forces central |
| `POMP_IGNORE_ENV` | `1`: `pomp_set_option` and `pomp_set_thread_limit` take precedence over the environment, and `OMP_THREAD_LIMIT` is not read at all (the host must set the limit). Read from the environment only |

## Host interface (`pomp.h`)

Every call returns one of the codes below, except `pomp_get_thread_limit` and the queries,
which return values.

```c
// Settings
int pomp_set_thread_limit(int limit);  // once per process; OMP_THREAD_LIMIT, if valid, wins
int pomp_get_thread_limit(void);       // initial thread included; 0 while no limit is set
int pomp_set_option(const char *name,  // any setting of "Environment", by its variable's
                    const char *value); //   name; only POMP_SPIN once the pool has opened

// The pool, in the order of a cycle
int pomp_open_pool(void);              // open the pool (once per cycle); needs a limit
int pomp_enter_pool(int slot);         // calling thread works in the pool, as slot in
                                       //   [1, limit-1]; returns once the pool closes
int pomp_close_pool(void);             // close the pool and release its threads; does NOT wait
int pomp_wait_pool_empty(void);        // optional: wait until released threads have returned

// Queries
int pomp_is_pool_open(void);           // nonzero while open
int pomp_pool_size(void);              // threads currently in the pool
int pomp_get_stats(unsigned long long *counts, int n); // diagnosis; 0 unless built with
                                       //   -DPOMP_STATS
```

`pomp_get_stats` copies up to `n` counters, in this order: regions forked with workers, idle
sleeps, team-wait sleeps, posts, posts at forks, ns spent posting, ns spent asleep, and root
descriptors allocated (see "Root descriptors").

### Initialization, and `pomp_open_pool`

There is no init call. pomp initializes itself on the first call of **any** entry point:
`pomp_enter_pool`, `pomp_set_thread_limit`, `pomp_get_thread_limit`, the other pool calls, and the
`__kmpc_*` calls of compiled code. Initialization reads the environment (`OMP_THREAD_LIMIT`,
`OMP_NUM_THREADS`, ...) once per process. Concurrent first calls are safe: one thread does it
and the others wait a few microseconds.

`pomp_open_pool` is a different thing: it runs once per *cycle*, as the counterpart of
`pomp_close_pool`. It lets threads enter, and lets parallel regions use them.

- The pool starts closed, whether or not `OMP_THREAD_LIMIT` is set: every cycle, the first
  one included, starts with `pomp_open_pool()`. Until the first open, regions run on one
  thread, and the first one to find the pool never opened warns on stderr.
- If `OMP_THREAD_LIMIT` is not set, the host calls `pomp_set_thread_limit(limit)` before
  `pomp_open_pool()`.
  Before the limit is set, the pool is closed: `pomp_open_pool` returns `POMP_ERR_LIMIT`,
  `pomp_enter_pool` returns `POMP_ERR_CLOSED` and regions run on one thread.
- If it is set but invalid, it is ignored with a warning on stderr, and pomp behaves as if it
  were not set. A host that wants to fail loudly, for example with a Java exception, checks
  `pomp_get_thread_limit() == 0` before its call to `pomp_set_thread_limit`.

So a thread may call `pomp_enter_pool` before anything else in the process: it initializes pomp
and enters the pool, or gets `POMP_ERR_CLOSED` if no limit is known yet.

| Code | Meaning |
|---|---|
| `POMP_OK` | Success. For `pomp_enter_pool`: the thread worked in the pool, was released by a close, and is back with the host |
| `POMP_ERR_LIMIT` | `pomp_set_thread_limit` with a `limit` not in `[1, 64]` while no limit is set, or different from the limit a previous call set; `pomp_open_pool` while no limit is set; `pomp_set_option` with an unknown name or an invalid value |
| `POMP_ERR_SLOT` | Slot out of range (`[1, limit-1]`), or another thread is in the pool as that slot and not released |
| `POMP_ERR_STATE` | Wrong caller or state: enter, close, open or wait from a pool thread or from inside a parallel region; `pomp_set_option` of a setting other than `POMP_SPIN` once the pool has been opened |
| `POMP_ERR_CLOSED` | Entry refused at once because the pool is closed (after a close, before the next open, or before any thread limit is known) |
| `POMP_ERR_NOMEM` | The pool's descriptors cannot be allocated when the limit is set; no limit is set, and a later call may retry |

### Design goal: the leader never waits for workers

The initial thread (the *leader*: a root that runs the model, slot 0 of its teams) is on the
critical path, so nothing in the protocol makes it wait for workers to arrive or to leave:

- **`pomp_close_pool` does not wait for anything.** It closes the pool with one store, claims
  every idle worker with one CAS on `busy`, marks them in `released` with one atomic OR, and
  returns. Spinning workers notice their bit and leave by themselves. Each worker that leaves
  wakes up to two sleeping ones, so sleepers are woken as a tree. The leader makes a system call
  only when *every* claimed worker is asleep, so that nobody else could start that chain, and
  then for one post only. Measured on macOS (`pomp_host_test latency`, 4 to 16 threads), the
  leader spends 0.1–0.25 µs in `pomp_close_pool` with spinning workers, and 1.2–1.5 µs when all
  of them sleep. The previous blocking version took 0.6–35 µs (spinning) and 10–45 µs (sleeping)
  at 4 to 16 threads.
- **Open does not wait for anything.** It checks that a limit is set, and reopens the pool with
  one store. Workers of the previous cycle that are still leaving keep their `busy` bit set until
  their slot is entered again, so a fork can never recruit them.
- **Forks never wait for pool threads.** A fork takes the workers that are idle at that moment,
  and a team may be smaller than requested. While the pool is closed, a fork runs on the leader
  alone.
- **Workers in a team when the close runs are released when that team ends.** This happens
  only if the close comes from another thread than the leader. The leader returns its team
  after the join, sees the pool closed, and claims and releases the workers itself. That costs
  one load of the pool's state per region when the pool is open.
- **Waiting is done by whoever needs it**, never by the leader. A pool thread whose slot is still
  being vacated by its previous thread waits for that thread itself. A host thread that wants
  all pool threads back calls `pomp_wait_pool_empty`.

### The thread limit

The thread limit is the most threads a team can have, the initial thread included. It also
bounds the slots: slot 0 is the initial thread of a team, a role any host thread outside the
pool takes in the teams it leads (each root has its own descriptor, see "Root descriptors"), and
the pool threads are slots `[1, limit−1]`. pomp treats it as OpenMP's `thread-limit-var`: **it
is set once per process and never changes**.

Where it comes from, in order:

1. **`OMP_THREAD_LIMIT`**, read at initialization, if it is set and in `[1, 64]`. It takes
   precedence: no call can override it.
2. Otherwise, **the first `pomp_set_thread_limit(limit)` with a `limit` in `[1, 64]`**. This
   stands in for the environment variable when the host cannot set one, and it is pomp's only
   addition to the standard here.

What `pomp_set_thread_limit(limit)` does (it never opens or closes the pool):

| Limit already set by | `limit` | Result |
|---|---|---|
| nothing | not in `[1, 64]` | `POMP_ERR_LIMIT`; no limit is set |
| nothing | in `[1, 64]` | The limit becomes `limit` |
| either | equal to the limit | `POMP_OK` |
| `OMP_THREAD_LIMIT` | anything else | `POMP_OK`, the limit unchanged, with one warning on stderr per process |
| `pomp_set_thread_limit` | anything else | `POMP_ERR_LIMIT`, the limit unchanged |

A host can therefore always call `pomp_set_thread_limit` with its own value: the environment
variable, if set, wins quietly. A second call with another value is a host bug, and is refused.
`pomp_open_pool()` takes no limit; it returns `POMP_ERR_LIMIT` while none is set.

**Relation to the OpenMP standard.** In OpenMP, `thread-limit-var` is initialized from
`OMP_THREAD_LIMIT` and there is no `omp_set_thread_limit`, so a program cannot change it. The
`thread_limit` clause and `omp_set_teams_thread_limit` concern teams constructs, which pomp
does not support. `pomp_close_pool` is the analogue of a **soft** `omp_pause_resource`: it gives
back the threads and keeps every setting, the limit included. As with a pause, nothing has to be
restored; the next region simply runs with whichever threads are in the pool then. The standard
does not reset or re-read ICVs on resume, so the limit after a pause is the one before it.
pomp follows that.

libomp itself is looser. Its re-initialization after a *hard* pause re-reads the whole
environment, so a program that changed `OMP_THREAD_LIMIT` with `setenv` would see the new value.
That is a side effect of an unfinished path in libomp (see
`astudy/analysis/lomp-standard.md` §16.5), not standard behavior, and pomp does not imitate it.

### Pool states

```
            pomp_open_pool()
   +----------------------------------+
   v                                  |
 OPEN ------ pomp_close_pool() --> CLOSED
   entries accepted                   entries return POMP_ERR_CLOSED at once
   forks recruit idle pool threads            forks run on the leader alone
                                        released threads return to the host
```

The pool starts CLOSED at initialization. Without a valid `OMP_THREAD_LIMIT`,
`pomp_set_thread_limit` must give the limit before the first `pomp_open_pool` (S12). A close
while CLOSED releases any pool thread claimed since the last close (normally none). An open
while OPEN does nothing.

No thread stays in the pool across a close. `pomp_close_pool` releases every idle thread at
once, and a thread busy in a team is released by its leader at that team's join. A thread still
leaving when the next `pomp_open_pool` runs keeps its `busy` bit, so no fork in the new cycle
recruits it, and it returns to the host normally. A thread joins a cycle only by calling
`pomp_enter_pool` after that cycle's open (S6).

### Supported scenarios

Each scenario has a test in `test/pomp_host_test.c`. All of them pass at thread limits 2, 3, 4,
8 and 16, with `POMP_SPIN` 0 and `auto` and both barrier kinds, with the `__thread` and the
pthread-key builds; `run_z.sh` also runs limit 17 and `POMP_SPIN` 5 and 200000. They run clean
under ThreadSanitizer.

**S1. Fresh threads per inference, killed after use** (`fresh`).

```
open; start N-1 threads, each pomp_enter_pool(i); wait until pomp_pool_size() == N-1;
run the model; close; join (or let exit) the N-1 threads
```

Once `pomp_close_pool` has returned, every pool thread is guaranteed to return from
`pomp_enter_pool` without anything else being called. Joining them is therefore safe at once,
and it costs the joining thread, not pomp. A pool thread may exit as soon as it returns: pomp
keeps no state on a thread that only entered the pool. (A host thread that also ran regions has
a root descriptor, which pomp reuses once the thread exits; see "Root descriptors".)

**S2. A host thread pool, reused across inferences** (`reuse`).

```
per inference: open; each pool thread calls pomp_enter_pool(i) again; run; close
```

A thread that returned from `pomp_enter_pool` is indistinguishable from one that never
called it. It may enter again as the same slot or another one, or act as the leader (S5).
`pomp_wait_pool_empty()` lets the host wait for its threads, from any thread that is not a
pool thread. It is never required for correctness.

**S3. Server mode: many inferences per entry** (`server`).

```
open; N-1 threads enter once; run M inferences; close once
```

Between inferences the workers stay in the pool: they spin for their adaptive spin
(`POMP_SPIN`) and then sleep, with no cost to the leader beyond the fork. Closing after every K inferences is S2.

**S4. Entries racing with close, and late entries** (`late`).

A thread may call `pomp_enter_pool` at any time relative to a close:

- **Before** the close reads `busy`: the close claims it, and it returns `POMP_OK`.
- **During** the close, between its own two steps: the close shuts the gate *before* claiming. A
  pool thread clears its busy bit *before* reading the gate. Either the pool thread sees the
  gate closed, takes its bit back and returns `POMP_ERR_CLOSED` (counted nowhere), or the close
  sees the bit and claims it. Both orders were exercised: the test sees both outcomes.
- **After** the close, before the next open: `POMP_ERR_CLOSED` at once.

In every case the pool thread returns to the host without anyone waiting for it, and
`pomp_pool_size()` is 0 once those returns have completed. No entry can stay stuck in pomp
across a close. This was the deadlock risk of the previous version.

**S5. Threads changing slots and roles** (`swap`).

A host thread may enter as slot 3 in one cycle and as slot 5 in the next, and a thread that
entered before may later be the leader. The only rule is one thread per slot at a time.

**S6. The next cycle starting while the last one is still leaving** (`overlap`).

```
open; enter; run; close; open; enter; run; ...   (no wait in between)
```

Workers of the previous cycle that have not left yet are never recruited, because their
`busy` bit stays set. A host thread entering again to a slot whose previous pool thread is still
leaving waits, in its own thread, for that pool thread to finish. Regions in the new cycle use
whoever has arrived.

**S7. Close while every worker sleeps** (`sleepers`, run with `POMP_SPIN=0`).

The leader posts to one sleeper, which wakes two more as it leaves, and so on, so every pool thread
returns. The leader pays one post.

**S8. Misuse** (`errors`).

- Slot 0 or a slot `>= limit`: `POMP_ERR_SLOT`.
- A slot held by a pool thread that has not been released: `POMP_ERR_SLOT`.
- Enter, close, open or wait from inside a parallel region: `POMP_ERR_STATE`. The same
  holds from a pool thread, which is always inside pomp.
- `pomp_set_thread_limit` with a limit out of range, or different from the one a previous call
  set: `POMP_ERR_LIMIT`.

**S9. Several host threads running regions at once** (`tworoots`).

Every host thread that is not in the pool becomes a *root* at its first region (fork,
serialized region or pushed team size), with a descriptor of its own: slot 0 of the teams it
leads. So roots share nothing but the pool. Each region gets the idle pool threads it can, up to
the limit, and its members see their own team in `__kmpc_barrier`, the loop init and the
`omp_*` queries, whatever the other roots do meanwhile. Together, roots may thus run more threads
than the limit, as OpenMP's contention groups do. The test runs 2 to 4 roots at once, with
barriers, nested and serialized regions, and pushed team sizes.

**S10. Nesting** (`nest`, with `OMP_MAX_ACTIVE_LEVELS=2`).

Members of a team continue their leader's nesting: they start at its depth and active level. So
the members of a level-2 team are at level 2, and a third parallel level runs on one thread, as in
libomp. As in libomp too, a level-2 team that got no worker is inactive, so a region nested in it
is only the second active level and may get workers; with fewer than 4 threads this happens. Beyond
8 nested levels, regions run on one thread and are only counted.

**S11. Waits that sleep** (`sleepjoin`, run with `POMP_SPIN=0`).

A slow member makes the leader sleep at the join and the other members sleep in both barrier
kinds; every one is woken by the thread it waits for.

**S12. No `OMP_THREAD_LIMIT`** (`nolimit`, run without it).

The pool starts closed, opens return `POMP_ERR_LIMIT`, entries return `POMP_ERR_CLOSED` and
regions run on one thread, until `pomp_set_thread_limit(limit)` sets the limit. The limit is then
fixed: the same value succeeds, another one returns `POMP_ERR_LIMIT`. Setting it does not open
the pool; `pomp_open_pool()` does.

**S13. Inner teams taking each other's workers** (`steal`, with `OMP_MAX_ACTIVE_LEVELS=2`).

Level-1 and level-2 teams change size every cycle, so inner teams take workers outside their
group, and workers move from one inner team to another. Every member must run the right work
with a distinct tid.

**S14. A limit set while regions run** (`lateopen`, run without `OMP_THREAD_LIMIT`).

Host threads run nested serialized regions and regions before any limit is set, and one of them
is inside a region while another thread sets the limit and opens the pool. Until then every
region runs on its root alone, at the right level; afterwards regions may use the pool.

**S15. Workers that leave and enter again** (`rejoin`, run with `POMP_SPIN=0`).

Pool threads leave and enter again as the same slots, many times, while the leader's regions
make them sleep in joins and barriers and be posted to. A slot's sleep is initialized by its
first entry only: a later entry finds it ready, and a waker still posting to the previous
occupant never finds it re-initialized.

**S16. Settings from code** (`options`, run without `OMP_THREAD_LIMIT`; also with
`OMP_NUM_THREADS` and `POMP_SPIN` set, and with `POMP_IGNORE_ENV=1`).

`pomp_set_option` refuses unknown names and invalid values, sets the limit as
`pomp_set_thread_limit` does, and changes every setting before the first open. After it, only
`POMP_SPIN` may change, while regions run; the others return `POMP_ERR_STATE`. With the
environment variable set, calls succeed and change nothing, unless `POMP_IGNORE_ENV=1`.

**S17. Host threads that come and go** (`rootchurn`).

Each host thread runs a few regions and ends. Its root descriptor is reused by the next host
thread: no more are allocated than roots run at once (counted with `-DPOMP_STATS`).

**S18. Reuse while a worker may still wake the last owner** (`latepost`, run with
`POMP_SPIN=0`).

A host thread runs one region and ends at once, while its workers may still be waking it; the
next host thread takes its descriptor and runs regions that sleep in their joins and barriers.
Such a late wake-up must only make the new owner re-check (see "Root descriptors").

**S19. A pushed size first, and a root in the pool** (`pushfirst`).

`__kmpc_push_num_threads` as the first call of a host thread applies to its first region only.
A host thread that has run regions may enter the pool, and runs regions again after it leaves.

### Rules for the host

1. Call close and open from a host thread outside the pool and outside parallel regions,
   normally the leader. A close while a leader is in a region is safe: the team's workers are
   released when it ends (see "Design goal" above). An open races only with the pool's state,
   so a fork in progress may or may not see it.
2. One thread per slot at a time. Using fixed slot numbers per host thread is simplest.
3. **Never kill a thread while it is inside `pomp_enter_pool`.** It may be a member of a team
   at that moment, and the leader would wait for it forever. A thread is outside once
   `pomp_enter_pool` has returned. This is the one way to deadlock pomp from outside, and
   pomp cannot detect it.
4. To have exactly N-1 workers in the first region, wait until `pomp_pool_size()` reaches
   N-1 after entering. Without that, the first regions run with whoever has arrived, which is
   correct but slower.
5. Any number of host threads may run regions at once (S9). They share the pool threads: a
   region gets those that are idle, so with several roots each gets fewer.
6. Never call `pomp_enter_pool`, `pomp_close_pool`, `pomp_open_pool` or
   `pomp_wait_pool_empty` from inside a parallel region; they return `POMP_ERR_STATE`.

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
which is one store to the thread's descriptor. A combined `#pragma omp parallel for`, which zDNNx
uses, emits the same calls minus the barrier, because the join already synchronizes.

Cost of each step, where *T* is the number of threads in the region (leader included) and *W*
is the number of those workers that are asleep rather than spinning:

| Step | Who | Cost | What it does |
|---|---|---|---|
| `__kmpc_fork_call`: recruit | leader | **O(1)** when the same team as last time is free (one CAS on `busy`); O(T) bit scan otherwise | Cached mask hit in the steady state |
| `__kmpc_fork_call`: describe | leader | O(T) | Writes the workers mask and copies `argc` pointers (plus an O(T) split of spare workers, only when 2 levels are enabled) |
| `__kmpc_fork_call`: wake | leader | **O(T)** atomic increments (one `go` epoch per worker) + 1 load of `sleeping`; + O(W) posts (system calls) | No system call when every worker spins |
| `__kmpc_global_thread_num` | each thread | O(1) | One `__thread` load (or `pthread_getspecific`) |
| `__kmpc_for_static_init_8u` | each thread | O(1) | A division and a few multiplies; no shared memory touched |
| loop body | each thread | O(n/T) | The work |
| `__kmpc_for_static_fini` | each thread | O(1) | Empty |
| `__kmpc_barrier`, central (small teams) | each thread | O(1) per thread, O(T) contention on one counter | The last arriving thread bumps the epoch and loads `sleepers`; the others spin on the epoch |
| `__kmpc_barrier`, tree (s390x, T ≥ 5) | each thread | O(1) per thread, **O(log₄ T)** depth, no contended line | Gather up a 4-ary tree, release down it; one load of `sleepers` per notify |
| `__kmpc_fork_call`: join | leader | **O(T) plain loads**, one per member's done flag, which overlap; then one atomic AND on `busy` and one load of the pool's state | No shared counter. Each member also loads `sleepers` once after its done flag |

So, per region:

- **Each worker does O(1) runtime work** besides its share of the loop: about five calls and no
  system call while everyone spins.
- **The leader does O(T) work**: it bumps one epoch per worker, and it reads one done flag per
  worker at the join. Neither involves a contended line, and the reads overlap. Everything else
  on the leader's side is O(1), including recruitment in the steady state.
- **The barrier** is O(log T) deep on s390x from 5 threads, and a central counter otherwise.
  The choice of each and its threshold come from the z17 measurements below.

### Cost compared with libomp (macOS)

These numbers predate the done-flag join and tree barrier described in "Measurements on z17"
below. On macOS that change is neutral for fork/join, and the central barrier is kept.

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
  on non-hybrid CPUs such as z. These timings were taken when pomp spun 200000 iterations
  before sleeping (now `POMP_SPIN`, adaptive by default).

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

### Measurements on z17 (Linux on Z), and the design choices they decided

Measured with `test/pomp_zbench.c` (see `test/ZBENCH.md`) on an IBM z17 (machine type 9175,
5.5 GHz) LPAR: 106 CPUs = 53 cores × SMT2, all in one drawer. The compiler was gcc 13.3 with
`-march=native`, and threads were pinned. Each figure is the median of 3 runs of 50000
regions or barriers.

**Machine costs.**

| | ns |
|---|---|
| Interlocked XOR / OR / AND / ADD (`LAXG`, `LAOG`, `LANG`, `LAAG`), uncontended | 1.27 |
| CAS (`CSG`) | 1.73 |
| seq_cst store or fence (`bcr 14,0`) | 0.36 |
| Cache-line transfer to an SMT sibling or another core on the same chip | 37–43 |
| Cache-line transfer to any other chip in the drawer, any book | 72–78 |

There are two tiers: the same chip (6 cores in this LPAR), and everything else in the drawer.
An SMT sibling is no closer than another core on the chip, and another book is no farther than
another chip in the same book. Running the same 8 threads as SMT pairs on one chip, instead of
one per core over two chips, halves the barrier cost (0.30 against 0.65 µs). Using that would
need the threads' topology, which z/OS may not expose, so it is on hold.

**Fork/join designs** (empty region, spinning workers, µs):

| Design | T=8 | 16 | 32 | 64 |
|---|---|---|---|---|
| A: epoch per worker + shared counter join (previous pomp) | 0.41 | 0.97 | 1.77 | 3.50 |
| B: one XOR activate word + counter join | 0.38 | 1.04 | 1.88 | 3.70 |
| **C: epoch or XOR wake-up + per-worker done flags (pomp now)** | **0.23** | **0.70** | **0.97** | 1.71 |
| D: XOR wake-up + 4-ary tree join | 0.27 | 0.78 | 1.06 | 1.67 |
| E: tree wake-up + tree join (libomp-like) | 0.26 | 0.74 | 1.01 | 1.65 |

pomp now combines its existing per-worker epoch wake-up with the done-flag join. That exact
pair was not in the benchmark. It is expected to match C because the two wake-ups, A and B,
cost the same on z; `pomp_zbench` on the new pomp would confirm it.

On z, the join dominates and the wake-up does not. Replacing the counter join with done flags
(A or B → C) saves 28% at 16 threads, 45% at 32 and 51% at 64. An XOR activate word alone (B)
gains nothing on z, although on macOS it cut 16-thread fork/join from 2.5 to 0.86 µs. The three
designs without a shared counter (C, D, E) are within 10% of each other; C is best up to 32
threads. With about 3.6 µs of work in the region, C saves 0.3 µs out of 4.6 at 16 threads.

**Barrier designs** (µs):

| Design | T=2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Central counter + epoch | **0.041** | **0.12** | 0.30 | 0.92 | 1.77 | 3.38 |
| 4-ary tree gather + broadcast word | 0.084 | 0.17 | 0.26 | 0.99 | 1.07 | 1.85 |
| **4-ary tree gather + tree release (pomp now, T ≥ 5 on s390x)** | 0.088 | 0.13 | **0.20** | **0.54** | **0.76** | **1.02** |

The tree in both directions is 41% faster at 16 threads, 57% at 32 and 70% at 64. The central
counter is best for 2–4 threads. Releasing through one broadcast word is poor, for the same
reason as design B above: a line that many threads spin on is expensive to update on z. The
run with one SMT thread per core gives the same ranking.

**Not measured:** threads spread over more than one drawer, sleeping workers, and pomp itself
(these are microbenchmarks of the patterns). The full outputs are with the author.

**On macOS (18-core Apple Silicon)** the same change was checked for regressions with
`pomp_test bench`. The done-flag join is neutral to slightly better: 0.37, 1.40 and 3.0–3.5 µs
fork/join at 4, 8 and 16 threads, against 0.40, 1.41 and 3.4 before. Adding sleeping team
waits left them unchanged within noise: 0.40, 1.44 and 3.1–3.5 µs, with barriers at 0.18, 0.47
and 0.74–0.94 µs. The tree barrier is
*slower* there at every size measured, up to 16 threads (1.9 against 0.75 µs at 16), so the
default keeps the central barrier on non-s390x targets. Other targets, such as many-core x86 or
AArch64 servers, were not measured.

## Language

`pomp.cpp` uses C++ for its classes and templates, and nothing that needs the C++ runtime. The
object file has the same undefined symbols as a C one (the C library and pthreads), no static
constructors, and links into C programs with the C driver: CMake builds it as a C source compiled
as C++, so nothing that links `libompruntime.a` gets a C++ library. The rules:

- **No C++ header**, enforced by `-nostdinc++`. Atomics are a small `Atomic<T>` class over the
  compiler's `__atomic` builtins, which `<stdatomic.h>` and `<atomic>` are both built on: the
  same instructions and memory model as C11. Mutexes and conditions are pthread's, behind
  `Mutex`, `Condition` and `ScopedLock<T>` classes.
- **No C++ runtime**: no exceptions, RTTI or thread-safe statics (`-fno-exceptions -fno-rtti
  -fno-threadsafe-statics`), no `new` or `delete`, no virtual functions, and no `thread_local` of
  a type with a constructor.
- **Constant initialization only.** Every global is `constinit` (or clang's
  `require_constant_initialization` in C++17) and trivially destructible, so nothing runs at
  load or exit. The descriptors come from `calloc`, as zero bytes are their initial state; no
  constructor runs. The pthread objects start zeroed and are initialized on first use.
- **Templates for the code that was macros**: the loop partitioning for the four IV types, the
  calls of outlined functions through their exact type (a table built at compile time), and the
  team waits, whose condition is a function object.

**One translation unit, in parts.** `pomp.cpp` holds the configuration, the entry points, and
the includes of its parts, in this order; all of them but the first are in its anonymous
namespace:

| File | Contents |
|---|---|
| `pompSupport.inc` | Code not specific to pomp: `POMP_CONSTINIT`, `fatal`, `relax`, `Atomic<T>`, `Mutex`, `Condition`, `ScopedLock<T>`, `Post`, and `invoke`, the call of an outlined function through its exact type |
| `pompIdentity.inc` | Slot sets, and the calling thread's descriptor (`Self`) |
| `pompSyncTypes.inc` | The types of the synchronization inside a team, which `Team` and `Thread` hold by value: `TeamWait`, `TeamJoin`, `TeamCentralBarrier`, `TeamTreeBarrier`, `MemberBarrier`, `TreeFlags`, `WorkerSync` |
| `pompTypes.inc` | The descriptors and the globals: the region stack, `Team`, `Thread`, `Pool`, `Settings`, `RootBlock`, `Roots` |
| `pompInit.inc` | Settings from the environment, the thread limit and the allocation of the descriptors |
| `pompPool.inc` | The idle wait, recruitment, and the host protocol of the pool |
| `pompSync.inc` | The code of the synchronization inside a team |
| `pompRegion.inc` | The region stack, the root descriptors, and fork/join |
| `pompLoop.inc` | Static loop partitioning |

As it is one translation unit, the inliner sees all of it, and the object code is the same as
with one file. The `.inc` files are not compiled on their own: copy them with `pomp.cpp`.

The tests are C programs and link with the C driver alone, so a C++ runtime symbol needed by
pomp would be a link error.

## Design

**Thread identity.** Each thread has one `__thread` pointer to its descriptor: its slot's
while it is in the pool, its root's otherwise (see "Root descriptors"), or null for a thread
that has run no region. The descriptor holds the slot (0 for a root), its region stack, and the
team size pushed by `num_threads` for its next fork.

Build with `-DPOMP_USE_PTHREAD_KEY` (CMake: `-DPOMP_USE_PTHREAD_KEY=ON`) to use a pthread key
instead, if `__thread` is unavailable in a static archive linked into `model.so`. The key is
created at initialization, and is not read before that: until then, every thread reads null.

Every entry point finds the calling thread from this pointer. The `gtid` argument of the
`__kmpc_*` entry points is ignored: it is only what the caller says, while the pointer cannot
name another thread (see "Input checks" below). Every root reports gtid 0.

**One word for the pool.** `busy` holds one bit per slot. A bit is clear iff that slot is a
idle worker in the pool. All slots start busy:
- A pool thread clears its own bit when it arrives. No team includes it yet, so there is no race.
- A leader recruits its whole team with one CAS (`busy |= pick`).
- After the join, the leader returns the team with one atomic AND.
- `pomp_close_pool` claims every idle bit the same way. Released slots stay busy until
  they are entered again.

**Descriptors, allocated with the limit.** When the thread limit is set, pomp allocates two
arrays with `calloc`, once per process, and never frees them, as the static arrays they replace
never were. No descriptor is O(limit²): the memory is linear in the limit.
- `threads[limit]`: per slot, one wake-up line with the wake-up epoch, the current assignment,
  the preferred workers for a level-2 team, and the sync words (done flag, tree barrier flags);
  the region stack and state, on lines of their own; and the sleep mutex and condition variable.
- `teams[limit]`: entry *i* is the team that pool thread *i* leads at level 2.

Entry 0 of both is unused: each root has its own (see "Root descriptors"). Zero bytes are the
initial state of both, so `calloc` is their whole construction, except for the sleep's mutex
and condition. Those are initialized once per slot, on demand, by the first thread that enters
as it, before it clears its `busy` bit. No other thread posts to a slot before that, as only
idle workers and team members are posted to. A slot's sleep is never initialized again, since a
waker that claimed a previous occupant's sleeping bit may still hold its mutex.

Until the limit is set, there are no pool descriptors, and every team size is 0: regions run on
their root alone, as serialized regions. The store of the limit publishes the arrays, and every
path that reads them first loads the limit with acquire, or follows a step that did: a fork
loads it before it recruits.

**Root descriptors.** A host thread that is not in the pool becomes a root at its first region
(or pushed team size): it takes a *root block*, one allocation with its descriptor, of slot 0,
and the two teams it leads: at level 1, and at level 2 as tid 0 of its level-1 team. In a team's
masks, slot 0 is that team's leader, which each team records at allocation. A root keeps its
block until it ends, so its next region touches nothing global but the pool, and roots never
share a descriptor.

Blocks come from a free list under a mutex, taken only at a thread's first region. A pthread
key, set once per root, has a destructor that returns the block to the free list when the thread
ends; it is a pthread destructor, as a C++ `thread_local` destructor would need the C++ runtime.
Blocks are never freed. A worker that has stored its `done` may still wake its leader, whose
thread may have ended and whose block may have a new owner: that wake-up reads only fields set
at allocation, and at worst posts to the new owner, which only makes it re-check. So a block is
reused whole, its teams with it, and its post is never cleared. A thread that ends inside a
region (`pthread_exit`) keeps its block, as a team may still use it. On z/OS, the destructor
was checked to run for threads that return, call `pthread_exit` or are detached.

A team descriptor holds:
- the outlined function and arguments;
- the team size, and the leader's depth and level, which members start from;
- the mask of its workers: tid 0 is the leader, and tid *t* > 0 the *t*-th worker in increasing
  order of slots;
- the reuse cache;
- the central barrier, and the `sleepers` mask of members asleep in a team wait.

**Fork.**
1. A host thread without a descriptor becomes a root (see "Root descriptors").
2. Decide the team size: the pushed `num_threads`, otherwise the `OMP_NUM_THREADS` entry for
   the current active level, clamped to the limit.
3. Run inline as a team of 1 in any of these cases:
   - the team size is 1;
   - the thread is already at `OMP_MAX_ACTIVE_LEVELS`;
   - 8 regions already enclose it;
   - the pool is closed;
   - no worker is idle.

   This matches libomp's default for the zDNNx matmul nest.
4. Otherwise recruit. If the request matches the team's cached one and those same workers are
   all idle, the pick is O(1). Otherwise take the lowest idle bits, preferring the calling
   thread's preferred set.
5. Write the descriptor, with the workers mask. Then set each member's `team` and `tid`, and
   bump its `go` epoch.
6. Run as tid 0, and join: wait for each member's `done` flag. Then clear the team's bits, and
   if the pool was closed meanwhile, release the workers.

**Two levels.** When a level-1 team forks with `OMP_MAX_ACTIVE_LEVELS=2`, the leader splits the
remaining idle workers into one contiguous group per member, and stores each group in that
member's descriptor before waking it. Each member's level-2 fork prefers its own group. Only
roots lead level-1 teams, so one group per slot is enough; two roots that fork at once may give
a slot two groups in turn, which only costs cache reuse. Inner teams therefore do not compete
with each other, and their caches stay valid from one region to the next. For example, with 16 threads, a 4×4 split gives the same
4 + 4×3 assignment every time.

**Wake-up without losing a signal.** An idle worker spins (for its `POMP_SPIN` time) on two
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

**Close.** `pomp_close_pool` closes the gate, claims the idle workers in `busy`, ORs them into
`released`, and returns; it posts once only if all of them sleep. Workers in a team at that
moment are not idle. Their leader claims and releases them the same way once it returns them
after the join.

A released worker leaves in this order:
1. It sets its slot to LEAVING and clears its `released` bit.
2. It wakes up to two released sleepers.
3. It restores its per-thread pointer (to its root descriptor, if it has one) and decrements
   the pool size.
4. It sets its slot to FREE.

Its `busy` bit stays set until the slot is entered again, so no fork can recruit a leaving
worker.

**Entry and the gate.** A pool thread first checks its own state: it is refused if no limit is
set yet, if the slot is out of range, or if the thread is inside pomp. It then takes its slot
(FREE to IN_POOL; if the slot is LEAVING it yields until FREE), resets its descriptor, and
clears its `busy` bit. Only then does it read the pool's state. The close writes the state
before it reads `busy`. With sequentially consistent accesses, either the pool thread sees the gate closed
or the close sees the cleared bit. If the pool thread sees it closed, it sets its bit back with
an atomic OR. If the bit was still clear, nobody claimed the slot, and the pool thread backs out
with `POMP_ERR_CLOSED`. Otherwise a close (or a fork after a new open) claimed it in between,
and it serves as an ordinary worker.

**Per-member sync words.** Each thread has three sync words: `done` (the join), and `arrive`
and `release` (the tree barrier). Only members other than the leader use theirs: a leader
stores no `done`, has no parent in the tree, and only touches the words of other members. A
thread is such a member of at most one team at a time, as it leads any team it forks and a
level-1 member cannot be recruited at level 2. So it takes its words from team to team, and
members reach each other's through the team's workers mask. Each word is written by one
thread and read by one other.

The words share the thread's wake-up line with its epoch and assignment. They are used in
turn: the recruiter writes the assignment and the epoch only while the thread is idle, and the
team uses the sync words from the wake-up until the leader has read `done`, before which the
leader does not return the thread to the pool. A woken thread thus finds its assignment on the
line it was polling, and a leader that forks the same team again already holds the lines whose
`done` it last read. The region stack, written at every barrier, is on lines of its own, so it
does not disturb a leader polling `done`. That is 1 line per slot for synchronization, where
one line per member of each team would be one per team.

The values are unique ids, so that a value left in a line by an earlier region, barrier or
team never matches. A team has two sequences of ids, one for its joins and one for its tree
barriers, each seeded with the team's index, set at allocation: the leader's slot for a pool
thread's team, and 64 + 2*k and 65 + 2*k for the two teams of the *k*-th root block. Each new
id is the last one plus 2^16, which every index is below: 2^16 bounds the root blocks to about
32 700, and pomp aborts beyond. So every id of a team is its index modulo 2^16, two teams never
meet, and a team's ids only grow, also when its block is reused. Every wait tests for
equality, and nothing is ever reset. Ids are advanced before use, so no id is 0, which every
line holds at first. A line keeps its `done` until the leader has read it, as the leader
returns its workers to the pool only after the join.

**Barrier.** The compiler emits one after every worksharing loop. For the tree barrier, each
thread keeps the unique id of its current barrier in its region-stack entry. The id starts from
the id of the team's last barrier, the same for every member, and advances by 2^16 at each
barrier.

- *Central*, for teams below `POMP_TREE_BARRIER`: each arriving thread increments the team's
  counter. The last one resets it, bumps the barrier epoch and wakes any member that sleeps; the
  others wait until the epoch changes.
- *Tree*, from `POMP_TREE_BARRIER` threads (default 5 on s390x): tid *t* is the parent of tids
  4*t*+1 … 4*t*+4. A thread waits until each child's `arrive` holds the barrier's id,
  publishes its own `arrive`, and waits for its `release`. Thread 0 does not wait for a
  release. Each thread then sets its children's `release`. The depth is ⌈log₄ T⌉. Each member
  finds the slots of its parent and children once, on entry to the region, from the workers
  mask, so that a barrier does no lookup.

**Join.** A member stores the region's unique join id into its `done`, with release ordering.
After that it touches only the team's `sleepers` word, to wake the leader if it sleeps, and the
leader's post; it reads the leader from the team before its store, a field set at allocation.
The leader waits for each member's `done` to hold that id. Then
it records the id of the team's last barrier for the next region, and owns the descriptor again. A
member's wake-up may still be in flight when the leader starts its next region on the team and
waits there. Like any post, it then only makes the leader re-check, and its bit is the one that
the leader's new wait set, so the post is consumed by that wait.

**Waits in a team sleep.** The join and both barriers wait with the same protocol as an idle
worker:
1. Spin, without yielding.
2. After its spin (`POMP_SPIN`: adaptive, or fixed µs), set the waiter's bit (its slot) in the
   team's `sleepers` word.
3. Fence, re-check the condition, and sleep on the waiter's own post.

A thread is in at most one wait at a time, so its slot's single post serves every kind of
wait. A waker publishes what it changed, fences, and then claims the sleeping bits of the
threads it releases, posting only to those whose bit it cleared. Like the idle wait, a post
means only "re-check". With every thread spinning, waking costs one extra load of a line that
stays shared, and no system call. These are the wakers:
- the last thread into a central barrier wakes the other members;
- in the tree barrier, a thread wakes its parent after its `arrive`, and its children after
  their `release`;
- each member wakes the leader after its `done`.

**Nesting.** Each thread keeps a region stack of 8 entries. Members of a team start at the
leader's depth and active level, which the team records, so nesting continues across threads.
Regions nested more than 8 deep run inline and are only counted. For them, `omp_get_level`
keeps counting, and `omp_get_thread_num`, `omp_get_num_threads`, the barrier and the loop init
see a team of one.

**Input checks: no index comes from the caller.** Every array access is bounded by pomp's own
state, never by an argument taken as is:
- **Slots.** `pomp_enter_pool(slot)` checks `1 <= slot < limit` before any use, and the limit
  is at most 64 (a static assertion ties `POMP_MAX_THREADS` to the 64-bit masks). Every other
  slot comes from the calling thread's own descriptor (set at allocation), or from a bit of a
  64-bit mask.
- **`gtid` arguments are ignored**, so a wrong `gtid` (negative, 64 or more, or another
  thread's) cannot index `threads`, read another thread's region stack, or join another
  team's barrier.
- **Thread limit.** `pomp_set_thread_limit` accepts only `[1, 64]` while no limit is set, and
  refuses any later value other than the limit set. `OMP_THREAD_LIMIT` out of range is ignored.
- **Team sizes.** `num_threads` is clamped to `[0, 64]` when pushed and to the limit at the
  fork; `OMP_NUM_THREADS` to `[1, 64]`. A team is at most the number of idle workers + 1.
- **Arguments.** `__kmpc_fork_call` aborts on `argc` outside `[0, 16]` before reading any.
- **Nesting.** The region stack holds 8 entries. Deeper regions are only counted, never stored.
  An `__kmpc_end_serialized_parallel` without a matching start aborts instead of popping below
  the stack.
- **Loop bounds.** The static loop init computes in the unsigned IV type, where arithmetic
  wraps, so no bound, increment or chunk (including the extremes of each type) makes it
  overflow. An increment of 0 aborts. A loop over all values of the type, whose trip count wraps
  to 0, runs on tid 0; libomp divides by zero there (chunked). An empty share never wraps
  around to the whole range. Otherwise the results are libomp's.

`test/` contains no test of this; it was checked with an adversarial driver under ASan and
UBSan (bad slots, limits, `gtid`s, `num_threads` and loop bounds), and against libomp at the
ends of every IV type.

**Failures and resources.** pomp allocates memory in two places, and never frees it:
- the pool's descriptors, once, when the thread limit is set. If that fails,
  `pomp_set_thread_limit` returns `POMP_ERR_NOMEM` and no limit is set (an `OMP_THREAD_LIMIT`
  that cannot be allocated is ignored, with a warning);
- one root descriptor per host thread running regions at once, at the first region of a thread
  when none is free (about 3 KB on s390x). If that fails, pomp aborts, as an entry point of
  compiled code cannot report an error.

Every pthread call is checked, and a failure aborts with the name of the call. Nothing else
aborts at run time, except:
- an outlined function with more than 16 arguments;
- a loop schedule other than static, or a loop increment of 0;
- an `__kmpc_end_serialized_parallel` without a matching `__kmpc_serialized_parallel`;
- more than about 32 700 root descriptors (see "Per-member sync words");
- a worker starting a region outside of a team, which is an internal error.

There is no `alloca` and no variable-length array. The largest frames on arm64 (`-O2`) are
`__kmpc_fork_call` at 272 bytes (16 argument slots), `__kmpc_barrier` at 144, and
`pomp_enter_pool` at 128. One nesting level (fork, invoke, outlined function's call) adds
about 400 bytes of pomp frames.

## Testing

Tests are in `test/`. Each test program builds against both pomp (`-DPOMP`, with threads
entering the pool standing in for the host) and libomp, so their outputs can be compared.

| Test | Checks | Result (macOS arm64, 18 cores) |
|---|---|---|
| `pomp_test static` | Owner of every iteration for 5 IV types × 13 trip counts × 4 steps × chunked, for every team size up to the limit | Digest identical to libomp at limits 1, 2, 3, 4, 5, 8, 16 |
| `pomp_test nested` | Team size at each level of the zDNNx 3-level `if` nest; `collapse(4)` | Identical to libomp with nesting off, `OMP_NUM_THREADS=4,4`, and `OMP_MAX_ACTIVE_LEVELS=2` (including partial 4/2 teams at limit 8) |
| `pomp_test stress N` | Many regions of varying size with a barrier | 200000 regions, no errors, at `POMP_SPIN` 200000, 50 and 0 (iterations, before it became µs); on z now 0, 5 and 200000 µs |
| `pomp_test barriers N` | 1–5 barrier pairs per region, team sizes 1 to the limit: after each barrier, every thread must see every other thread's write from before it | No errors at limits 1, 2, 3, 4, 5, 8, 16 and 17, with `POMP_TREE_BARRIER` 0, 2, 5 and 65; also under ThreadSanitizer |
| `pomp_test cycles N` | Enter, run 10 regions, close: the per-inference pattern | 20000 cycles, no errors; about 200 µs per cycle at limit 8, mostly `pthread_create`/`join` in the test host |
| `pomp_test bench`, `loop`, `nestbench N n1 n2` | Timings: empty fork/join and barrier; one loop region; an outer team of n1 whose members each fork n2 threads for a loop. `POMP_TEST_CPUS` pins thread t to the t-th CPU listed, for pomp and libomp alike | See `test/run_z.sh`, which runs them against libomp |
| `pomp_host_test <scenario>` | Host protocol scenarios S1–S19 above, and `latency` (leader time in `pomp_close_pool`) | All pass at limits 2, 3, 4, 8 and 16, `POMP_SPIN` 0 and `auto`, both barrier kinds, both thread-slot variants (724 runs); `latepost` also 3000 cycles at limits 2 to 16; no ThreadSanitizer reports. Limit 1 is not a supported test configuration: `late` and `swap` assume a slot 1 exists |
| `pomp_irbuilder_test` | Code from `-fopenmp-enable-irbuilder`, the OpenMPIRBuilder path MLIR uses, which calls `__kmpc_global_thread_num` on workers | Correct at limits 1, 4, 18; also with `POMP_USE_PTHREAD_KEY` |
| `test/run_sweep.sh` | Builds both thread-slot variants and runs the tests above (except the timings) at limits 1 to 64, `POMP_TREE_BARRIER` 2, 5 and 65, `POMP_SPIN` 0 and 200000 (µs since the change; the 3228 runs below were with iterations); at 1 active level, and at 2 (`barriers2`, `lockstep`, `nested`, `nestbench` shapes 2x2 to 8x8, host `nest` and `steal`) | 3228 runs, no failures |
| ThreadSanitizer | `static`, `nested` (both nesting modes), `stress`, `barriers`, `cycles`, and all of `pomp_host_test`, with both barrier kinds and `POMP_SPIN` 0 and 30 (iterations, before it became µs) | No reports |

Measured costs against libomp are in "Cost compared with libomp" in the example section above.

## Building outside onnx-mlir

```sh
clang++ -O2 -std=c++17 -nostdinc++ -fno-exceptions -fno-rtti -fno-threadsafe-statics -c pomp.cpp
clang -O2 -fopenmp -DPOMP -c test/pomp_test.c && clang pomp_test.o pomp.o -o pomp_test
OMP_THREAD_LIMIT=8 ./pomp_test static
clang -O2 test/pomp_host_test.c pomp.o -o pomp_host_test
OMP_THREAD_LIMIT=8 ./pomp_host_test reuse
./pomp_host_test nolimit 100 8       # without OMP_THREAD_LIMIT: the limit comes from a call
```

On macOS with Apple clang, use `-Xclang -fopenmp -I$(brew --prefix libomp)/include` for
`-fopenmp`; no libomp is linked.

## Tried and not kept: yielding while spinning, as libomp does

libomp's spin waits call `sched_yield` once every 512 spins by default on Linux
(`KMP_USE_YIELD=1`, `KMP_INIT_WAIT`/`KMP_NEXT_WAIT` in `kmp.h`); pomp's idle workers never yielded,
and its team waits then yielded on every spin after 4096 (since removed: waits now spin for
`POMP_SPIN` µs without yielding). As pomp loses about 2.8× from pinned to
free threads where libomp loses nothing, a `POMP_YIELD=N` setting made every wait yield once
every N spins. Measured on z17 with `test/run_z.sh` (µs, median of 5 runs):

| Region overhead | pomp | N=512 | N=64 | libomp |
|---|---|---|---|---|
| T=2, free / pinned | 0.81 / 0.28 | 0.77 / 0.28 | 0.81 / 0.64 | 1.31 / 1.44 |
| T=4, free / pinned | 1.32 / 0.47 | 1.36 / 0.47 | 1.31 / 1.05 | 1.85 / 1.63 |
| T=8, free / pinned | 1.66 / 0.67 | 1.69 / 0.67 | 1.97 / 1.52 | 2.27 / 2.74 |
| T=16, free / pinned | 2.30 / 1.60 | 2.36 / 1.61 | 2.58 / 2.22 | 2.76 / 3.42 |
| 8x8 nested, free / pinned | 3.12 / 3.08 | 4.96 / 3.13 | 4.16 / 3.55 | 12.4 / 8.2 (hot teams) |

Yielding does not close the gap: free runs are unchanged at N=512 and no better at N=64, and
N=64 doubles the pinned cost, as a woken thread may find its CPU given away. The yield was
removed; the gap between free and pinned threads is a matter of placement, not of spinning.

## Tried and not kept: team reuse from the Blue Gene/Q runtime

Branch `pomp-bgq-attempt` applied two ideas from Eichenberger and O'Brien, "Experimenting with
low-overhead OpenMP runtime on IBM Blue Gene/Q" (IBM J. Res. & Dev. 57(1/2), 2013), to the fork:

- when a leader forks the same workers as last time, it skips writing their assignments (team,
  tid), with a per-team counter to detect workers taken by another team meanwhile;
- it wakes the whole team with one atomic XOR on a shared word, instead of one epoch per worker.

On macOS this made a 16-thread fork/join 5× faster (3.4 to 0.66 µs). On z17 it made no
difference at 2 to 16 threads, the team sizes onnx-mlir uses, and helped only at 64:

| Loop region, µs, z17 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| pomp (this branch) | 1.38 | 1.83 | 2.48 | 2.84 | 4.67 |
| pomp-bgq | 1.46 | 1.84 | 2.38 | 2.60 | 3.80 |
| libomp | 0.89 | 1.66 | 2.49 | 3.21 | 5.18 |

The shared wake-up word is the reason: on z, a line that one thread writes and many read is
expensive, as for the barrier's broadcast release above. `pomp_zbench` shows it: design F
(pomp's wake-up) beats C (the XOR word) up to 16 threads, 0.48 against 0.67 µs at 16. Skipping
the assignment writes helps a little, but not enough for its extra state, so the simpler fork is
kept. The libomp column used `OMP_PROC_BIND=close` with pomp unpinned; `test/run_z.sh` now
times both runtimes with the same placement.

## Open items

- **pomp itself not yet built or run on z/OS or Linux on Z.** The pattern microbenchmarks ran on
  z17 Linux (above). Still to confirm on z/OS: `__thread` in a static archive linked into a
  model `.so`. Atomics compile to single interlocked instructions on Linux on Z (gcc 13.3). The
  cache line is set to 256 bytes for s390x.
- **Topology-aware placement is on hold.** Packing a team onto one chip halves synchronization
  cost on z17, but z/OS may not tell us where the Java threads in the pool run.
- **No end-to-end test with an onnx-mlir model yet.** The installed `onnx-mlir` binary currently
  fails to load because of a Homebrew abseil upgrade.
- **zDNNx still compiles to `_4u`, `_8` and the serialized pair.** pomp supports them, so the
  source simplification in `openmp-calls-in-om.md` §4.2 is optional.
