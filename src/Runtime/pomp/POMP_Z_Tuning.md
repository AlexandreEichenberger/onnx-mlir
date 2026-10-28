# Tuning pomp on Linux on Z

The current best understanding of pomp's performance on Z, organized by topic. How to run the
tools: [test/ZBENCH.md](test/ZBENCH.md).

> **Instructions for updating this file with new measurements.** This is a living document, not
> a log. Please add a summary of any new experiments below. When given new output (usually a
> `run_z.sh` log):
>
> 1. Integrate the results into the topic section they belong to. If a section is marked
>    "not yet run on Z", fill it in with the Z results and its conclusions.
> 2. Replace any conclusion the new data contradicts; do not append corrections or keep the
>    history of what was believed before. Remove results that no longer matter.
> 3. A new topic gets a new section, as short as the others: the question, the setup, the key
>    numbers in a small table, the conclusions.
> 4. Define every new placement, design or wait policy under "Terms".
> 5. Update "Open items": remove the ones done, add the new questions.
> 6. Keep the commit list under "Setup" current. Note a number that moves by more than about 5%
>    between runs only if it changes a conclusion.
>
> "Section N" means a section of this file; a section of `pomp_zbench`'s output is
> "zbench part N".

## Summary

Numbers from the latest runs, µs. **Default** means each runtime as an app gets it, with no
environment set: threads `free` (Linux places them); libomp with `KMP_BLOCKTIME=200`, its
default on Z; pomp with `POMP_SPIN=auto`, its default since commit f99008082 (an adaptive spin,
section 6). **Best** means the best setting found for that scenario; the setting is named.

| Scenario | libomp default | libomp best (setting) | pomp default (`auto`) | pomp best (setting) |
|---|---|---|---|---|
| Region overhead, T=2 | 1.38–1.52 | 0.70 (`chip-free`) | 0.93 | 0.26–0.28 (`pinned`); 0.33 unpinned (`chip-free`) |
| Region overhead, T=4 | 1.76–1.95 | 0.82 (`chip1-free`) | 1.34 | 0.49 (`chip-free`) |
| Region overhead, T=8 | 2.38–2.47 | 1.24 (`chip-free`) | 1.83 | 0.64–0.76 (`pinned`); 0.83 unpinned (`chip-free`) |
| Region overhead, T=16 | 2.61–2.83 | 2.40–2.49 (`cores-close`) | 2.29 | 1.64 (`pinned`) |
| Barrier, T=2 / 4 / 8 / 16 | 0.40 / 0.51 / 0.63 / 0.72 | 0.15 / 0.17 / 0.35 / 0.72 | 0.13 / 0.56 / 0.72 / 0.67 | 0.05 / 0.16 / 0.31 / 0.51 |
| Nested 8x8, hot teams | 11–15 | 8.0 (`pinned`) | 3.4 | 3.1–3.3 (`free`, `pinned`) |
| Waiting next to a computing SMT sibling, 200 µs of work, barrier (excess) | 113 | 25 (`KMP_BLOCKTIME=0`) | **26** | 26 (`auto`, or `POMP_SPIN=0`) |
| Same, short work (20 µs) | 11 | 4.7 (`KMP_BLOCKTIME=0`) | 6.2 | 4.4 (`POMP_SPIN=0`) |
| Waiting on its own core, 20 µs of work, barrier (excess) | 0.8 | 0.8 (default) | 0.8 | 0.8 (`auto`) |

What this shows:
- **pomp's adaptive spin gets each case within a few µs of its best fixed setting,** where every
  fixed spin, pomp's or libomp's, loses badly in one of them: libomp's default costs 113 µs next
  to a computing sibling; `KMP_BLOCKTIME=0` (and pomp's earlier 5 µs) make back-to-back T=16 and
  nested regions 2–10x slower. The one case it leaves: short uneven waits next to a sibling
  (20 µs of work), 6–9 µs against 4–6 sleeping at once (section 6).

What an app should do on Z:
- **Keep the team on one chip** when it fits (12 CPUs here): `taskset`, a cpuset, or pinning.
  Left free, Linux spreads threads over chips, which doubles the cost of every handoff. This
  roughly halves libomp's overhead and is worth 2–3x for pomp at T ≤ 8.
- **pomp: keep the default `POMP_SPIN=auto`.** **libomp: choose `KMP_BLOCKTIME` for the workload:**
  `0` if threads that wait share a core with threads that still compute (uneven work, two
  threads per core); the default if regions follow each other with little work in between, or
  are nested.
- libomp's own binding: use `OMP_PLACES=cores OMP_PROC_BIND=close`, never `spread` (it puts
  threads on other chips on purpose).
- At each runtime's best setting, pomp's overhead is 1.5–2.7x lower than libomp's; on real work
  with uneven threads, the wait policy matters more than that difference.

## Setup

- Machine: Linux on Z LPAR, 106 CPUs (0–105), all in one drawer.
- Tools: `test/run_z.sh` builds pomp from source, checks it against libomp (552 stress runs, no
  failure; digests match libomp's), times both runtimes, then runs `test/pomp_zbench`.
- libomp: onnx-mlir's `libompruntime.a`, always built as Release, so the comparison is fair.
  Its wait policy in all timings: `KMP_BLOCKTIME=200` (spin 200 ms, then sleep).
- Numbers come from runs at pomp commits fbae69c24, 7d6fe3917, aa1111068 and 4f8cb8704, with
  `POMP_SPIN` counting iterations (default 200000), and 802336526, 42a3b0b8e and
  c8b0916b8, with `POMP_SPIN` in µs (default 5), and f99008082, with `POMP_SPIN=auto` (default). pomp's numbers are
  stable within about 5% across runs; libomp's free (unpinned) numbers vary by up to 15%.
- Times are medians, in µs or ns.

## Terms

**Leader** (OpenMP's primary thread): the thread that starts a parallel region. It wakes the
other threads of the team, its **workers**, then waits for them at the **join**, the end of the
region. **T**: the number of threads in the team, the leader included. **Overhead**: the time of
a region with one static loop, minus one thread's share of that loop run alone.

Machine: a **core** runs two hardware threads (**SMT**); each is a CPU to Linux, and the two are
**SMT siblings**. A **chip** here has 6 cores, 12 CPUs; CPUs 0–11 are one chip. A cache line
moves between two CPUs of one chip in about 38 ns, and to any other chip in 72–78 ns (section 1).

Placements, the CPUs a team's threads run on:

| Name | Pinned? | CPUs |
|---|---|---|
| `free` | no | any; Linux places them |
| `pinned` | yes, thread *t* on the *t*-th CPU | 0, 1, 2, ...: two threads per core, chip 0 first |
| `pinned1` | yes | one CPU per core, 0, 2, 4, ...: one thread per core; T=8 and 16 reach a second chip |
| `chip-free` | no | the process limited (`taskset -c 0-11`) to the leader's chip; Linux places threads in it |
| `chip1-free` | no | as `chip-free`, one CPU per core of the chip (`taskset -c 0,2,...,10`) |
| `cores-close` | by libomp | libomp only: `OMP_PLACES=cores OMP_PROC_BIND=close`, one core per thread, packed together |
| `cores-spread` | by libomp | libomp only: `OMP_PLACES=cores OMP_PROC_BIND=spread`, one core per thread, spread over the machine |

Wait policies: a waiting thread spins (polls a flag) for a while, then sleeps until woken.
**`POMP_SPIN`** (pomp): `auto` (default) adapts each thread's spin, separately for waiting for
work and for team waits, between 2 and 100 µs: after a wait of *d* µs, the next spin covers 2·*d*
if *d* ≤ 100, and halves (to at least 2) if *d* > 100; `auto:M` sets the ceiling to M. A number
N spins exactly N µs (0 sleeps at once). Never yields. From commit 802336526 to the adaptive
change, the default was a fixed 5 µs. Up to commit 4f8cb8704 it counted iterations (default 200000), and barrier and join
waits called `sched_yield` after 4096 of them; numbers from those runs say "iterations". **`KMP_BLOCKTIME`**
(libomp): time spent spinning before sleeping (`200` = 200 ms, `20us`, `0` = sleep at once).

Fork/join designs timed in `pomp_zbench` (zbench parts 4 and 6). All workers spin while waiting.

| Design | Wake-up | Join |
|---|---|---|
| E | 4-ary tree: each thread wakes up to 4 children | 4-ary tree: each thread waits for up to 4 children's done flags, then sets its own (close to libomp's design) |
| F | per worker, the leader writes an assignment, then adds 1 to the worker's epoch word, on another line | each worker sets its own done flag, on its own line; the leader polls them in order |
| I | as F, with each worker's epoch, assignment and done flag on one line: **pomp's design** | as F, on that line |
| L | as I, waking the workers on other chips first | as I |

## 1. The machine

| Operation (zbench part 1, uncontended) | ns |
|---|---|
| load, store (seq_cst included), seq_cst fence | 0.24 |
| `fetch_add`, `fetch_or`, `fetch_and`, `fetch_xor` | 1.27 |
| compare-and-swap | 1.66 |
| exchange | 1.81 |

| One-way line transfer from CPU 0 (zbench part 2) | ns |
|---|---|
| SMT sibling (CPU 1) | 42 |
| another core of the chip (CPUs 2–11) | 36–40 |
| another chip, any book (CPUs 12–105) | 72–78 |

- Atomics cost nothing that matters; what counts is how many lines move, and how far.
- There are two distances: the same chip, and anywhere else. An SMT sibling is no closer than
  another core of the chip.

## 2. pomp against libomp

Each runtime at its own best placement, among all those under Terms:

| T | libomp overhead (place) | pomp overhead (place) | ratio | libomp barrier | pomp barrier | ratio |
|---|---|---|---|---|---|---|
| 2 | 0.672 (`chip1-free`) | 0.267 (`pinned`) | 2.5x | 0.15 | 0.05 | 3.0x |
| 4 | 0.832 (`chip-free`) | 0.470 (`pinned`) | 1.8x | 0.17 | 0.16 | 1.1x |
| 8 | 1.246 (`chip-free`) | 0.647 (`pinned`) | 1.9x | 0.35 | 0.21 | 1.7x |
| 16 | 2.337 (`cores-close`) | 1.541 (`pinned`) | 1.5x | 0.66 | 0.42 | 1.6x |

- **pomp is 1.5–2.5x faster per region and 1.1–3.0x per barrier.** Quote these.
- pomp `chip-free`, its best unpinned placement, is within 0–20% of `pinned` (T=2: 0.319, T=4:
  0.474, T=8: 0.734 µs).
- Nested regions (an outer team whose members each fork an inner team), against libomp keeping
  its inner teams (hot teams), `free`: pomp 1.7–3.7x faster for shapes 4x4 to 8x8. libomp by
  default builds every inner team again in each region; pomp is 10–35x faster than that. Nested
  runs have not been timed at each runtime's best placement yet.
- Do not quote the `pinned` ratios (up to 5.6x per region, 11x per barrier): they put libomp's
  worker on the leader's SMT sibling, its worst placement within a chip.

## 3. Placement

T=2, both threads pinned, the worker by distance from the leader (overhead, µs):

| Worker on | pomp | libomp |
|---|---|---|
| SMT sibling (CPU 1) | 0.273 | 1.467 |
| another core of the chip (CPU 2) | 0.325 | 1.123 |
| another chip (CPU 12) | 0.837 | 1.590 |

- **Distance dominates, for both runtimes:** a worker on another chip costs about 0.5 µs at
  T=2. pomp's time is mostly line transfers, so this is its largest single cost.
- **Linux spreads free threads one per core and onto other chips** (`pomp_test place`): at T=2
  the worker is on another chip in 100% of regions, at T=16 81–89% of workers are. It neither
  migrates nor preempts them, and the cost is steady (p99 within 20% of p50); all CPUs are
  horizontally polarized. That is why `free` is slow for pomp (T=2: 0.84–0.88 µs).
- **Confining the process to the leader's chip recovers it without pinning:** `chip-free`
  brings pomp within 0–20% of `pinned`, and is libomp's best placement for T=2 to 8, about twice
  as fast as `free`.
- Sharing a core costs libomp 0.34 µs at T=2, and pomp 0.05.
- libomp's own binding: `cores-close` is its best at T=16; `cores-spread` is as slow as `free`,
  since it puts threads on other chips on purpose. At T ≤ 8 `cores-close` is a little slower
  than `chip-free` (T=8: 1.62 against 1.25 µs); it started the team on CPU 14, on another chip
  than CPU 0.
- Open: libomp with its threads pinned by the test (`pthread_setaffinity_np`) on CPUs 0 and 2 is
  1.10–1.12 µs; bound by itself, or confined by `taskset`, to the same kind of CPUs it is
  0.67–0.71 µs. pomp shows no such gap.

## 4. Fork/join design

Region time in µs (zbench part 4, `pinned`, empty body unless noted):

| Design | T=4 | 8 | 16 | 16, body of 2000 iterations |
|---|---|---|---|---|
| E | 0.141 | 0.264–0.279 | 0.67–0.71 | 4.25–4.28 |
| F | 0.157 | 0.243 | 0.68–0.72 | 4.28–4.35 |
| I (pomp) | 0.096 | 0.157–0.160 | 0.38–0.41 | 4.04 |
| L | 0.089–0.094 | 0.160–0.165 | **0.29–0.30** | **3.85–3.98** |

- **pomp's layout (I) is the best of the designs tried, except L.** Keeping a worker's epoch,
  assignment and done flag on one line makes its wake-up one line transfer per worker; at T=16
  the leader's wake loop takes 139 ns, against 467 for F's two lines.
- **Waking the workers on other chips first (L) cuts the T=16 region by 25–27%.** Their 75 ns
  transfers then overlap the near ones instead of coming last. At T ≤ 8 every worker is on the
  leader's chip and L equals I. This is the one fork/join change worth making in pomp.
- The wake loop's instructions alone take 3–24 ns: its cost is line transfers.
- The order in which the leader polls the done flags does not matter measurably: polling in the
  worst order cost 15–30 ns, and the join after the last worker is done is mostly one transfer.
**Tried and rejected** (pinned, T=16 unless noted; don't try these again without a new reason):

| Idea | Z result | Why it fails |
|---|---|---|
| Write the epoch with a release store instead of an atomic add | 0.412 against 0.413 µs | No gain: an interlocked add costs 1.3 ns, the line transfer dominates. In pomp it also needs a seq_cst fence, or a sleeping worker can miss its wake-up |
| Prefetch every worker's line for writing, then wake | 0.427 against 0.412 µs | Worse on Z; it halved the wake loop on the Mac only |
| Done flags as bytes, 8 per line | 0.86 against 0.68 µs | The 8 workers sharing a line contend for it |
| Leader sweeps the done flags not yet set, instead of in order | ≤ 30 ns gain | Only in the worst wake order; the last worker done is usually the last polled |
| One shared wake word that every worker polls (one XOR per region) | 0.69–0.72 against 0.68 µs | No faster than a word per worker on Z, unlike the Mac |
| Shared counter for the join | 0.95–1.06 against 0.68 µs | Arrivals serialize on one line |
| 4-ary tree wake-up and join (E, libomp-like) | 0.67–0.71 µs | About F, behind I (0.38–0.41) |
| `sched_yield` while spinning (SMT mitigation, zbench part 7) | sibling 1.21–1.40x slower | No better than plain spinning |
| 64-iteration backoff between polls (zbench part 7) | sibling 1.16–1.21x slower | Recovers only about a third of the slowdown |
| libomp `OMP_PLACES=cores OMP_PROC_BIND=spread` | as slow as `free` | Spreads the team over chips on purpose |

## 5. SMT interference

zbench part 7: one thread computes on CPU 0 while a partner waits; the time against the thread
alone.

| Partner on | asleep | spins (as pomp) | spins and yields (as libomp) | spins with a backoff |
|---|---|---|---|---|
| SMT sibling, float loop | 1.00x | 1.29x | 1.40x | 1.21x |
| SMT sibling, integer chain | 1.00x | 1.30x | 1.21x | 1.16x |
| another core or chip | 1.00x | 1.00–1.02x | 1.00–1.02x | 1.00–1.02x |

- **A waiting thread that spins slows its SMT sibling by about 30%.** On any other core it costs
  nothing, and asleep it costs nothing.
- Yielding does not help; a short backoff between polls wins back only a third.
- This explains why libomp's work is 1.45x slower `pinned` than pomp's: libomp's workers spin
  (200 ms) on the leader's SMT sibling while it computes; pomp's have gone to sleep.
- It applies to pomp whenever a waiting thread shares a core with one that computes, for example
  a worker that finishes first and waits in a barrier. Empty-region timings miss it.

## 6. Spin or sleep: waiting next to a computing thread

The question: a thread that waits at a join or a barrier can spin, which saves the 1–4 µs of a
wake-up, or sleep. Section 5 shows that a spinning thread slows its SMT sibling. With real work,
which costs more? `pomp_test imbal` (run by `run_z.sh`) measures it:

- In each region, even tids run LONG passes of work and odd tids SHORT ones (one pass scales 1024
  floats, about 0.08 µs): 250/25 passes (long work about 20 µs alone) and 2500/250 (about
  200 µs). Then all meet at the join (`join`), with the leader short (`join,ldr-short`: the
  leader waits next to a long worker), or at 10 barriers per region (`barrier`).
- `sibling`: tids 2k and 2k+1 on the two CPUs of one core, so each long thread computes next to a
  waiting one. `own-core`: one core per thread, the control. T=2 and 4, on the leader's chip.
- **Excess** = time per phase minus the long work run by one thread alone: the runtime's
  overhead plus the slowdown of the long work by its sibling.

Excess in µs, pomp with its old iteration-count spin (4f8cb8704), T=2 (T=4 is similar; libomp
in the last column for comparison):

| Place, work | sync | 200000 iterations (then default) | 20000 | 2000 | 0 | libomp 200 ms / 0 |
|---|---|---|---|---|---|---|
| sibling, 20 µs | join | 8.1 | 9.0 | 4.5 | 3.9 | 11.5 / 4.7 |
| sibling, 20 µs | barrier | 10.1 | 10.1 | 4.4 | 4.4 | 11.3 / 4.7 |
| sibling, 200 µs | join | 75 | 31 | 26 | 26 | 114 / 26 |
| sibling, 200 µs | join,ldr-short | 114 | 114 | 29 | 28 | 114 / 28 |
| sibling, 200 µs | barrier | 113 | 113 | 26 | 26 | 113 / 25 |
| own-core, 20 µs | join | 0.8 | 1.3 | 1.1 | 1.2 | 1.0 / 1.4 |
| own-core, 20 µs | join,ldr-short | 1.2 | 1.5 | 4.5 | 3.7 | 1.7 / 4.3 |
| own-core, 20 µs | barrier | 0.8 | 0.8 | 1.5 | 1.5 | 0.8 / 1.5 |
| own-core, 200 µs | join | 10.0 | 10.5 | 11.5 | 10.2 | 7.5 / 9.1 |

Conclusions:
- **Next to a computing sibling, spinning is by far the larger cost.** It slows the sibling for
  the whole wait: with 200 µs of work, 113 µs of excess spinning against 26 µs sleeping, so
  the phase takes 305 against 218 µs (40% longer). Even with 20 µs of work, sleeping wins
  (4 against 10 µs). That is 50–100 times the 1 µs difference in fork/join overhead between
  the runtimes, which is what the empty-region timings optimize.
- **On its own core, sleeping costs 1–3 µs per wait** (the wake-up through the kernel), and only
  for short waits: with 200 µs of work, the policies differ by about 1 µs.
- **Spinning for 200000 iterations, pomp's old default, spun through all of a 180 µs wait.**
  Counting iterations also made the spin last very differently in the two kinds of wait:
  barrier and join waits called `sched_yield` after 4096 iterations, making each later one a
  system call (likely the reason 20000 iterations slept within a worker's 180 µs wait, but not
  at a barrier's; not checked directly). Hence the change below, to a spin measured in time.
- **libomp behaves the same**: `KMP_BLOCKTIME=0` matches pomp's best; its default spins through
  the wait like pomp's. `KMP_BLOCKTIME=20us` gave exactly the 200 ms numbers (cause not
  checked).
- **pomp's wait now spins for a time** (commit 802336526): `POMP_SPIN` is microseconds of
  spinning, default 5, read from the TOD clock (`STCKF`) every 64 spins, in every wait (idle
  worker, join, barrier), with no `sched_yield`. Excess with it, µs (T=2; T=4 alike):

  | Place, work | sync | 200000 µs | 20 | 10 | 5 | 2 | 0 |
  |---|---|---|---|---|---|---|---|
  | sibling, 200 µs | barrier | 61 | 28 | 26 | 25 | 25 | 25 |
  | sibling, 200 µs | join,ldr-short | 86 | 31 | 28 | 27 | 27 | 27 |
  | sibling, 20 µs | barrier | 6.1 | 6.9 | 5.8 | 4.9 | 4.6 | 4.5 |
  | own-core, 20 µs | barrier | 0.9 | 0.8 | 1.3 | 1.3 | 1.3 | 1.5 |
  | own-core, 20 µs | join,ldr-short | 1.4 | 1.3 | 3.1 | 2.8 | 3.5 | 3.9 |

  On these uneven-work tests 5 µs is right: next to a sibling it is as good as sleeping at
  once, and on its own core it loses only 0.5–1.5 µs per wait to spinning 20 µs.
- **But 5 µs is too short for regions where nothing computes between waits.** At the 5 µs
  default, the one-level and nested timings got much worse (same runs, µs):

  | | 200000 iterations (4f8cb8704) | 5 µs (802336526) | libomp |
  |---|---|---|---|
  | T=16 overhead, free / pinned | 2.26 / 1.54 | 5.48 / 3.12 | 2.83 / 3.43 |
  | T=16 fork/join, free / pinned | 1.14 / 0.81 | 7.99 / 7.11 | 1.78 / 1.68 |
  | Nested 8x4, free / pinned | 2.62 / 2.03 | 10.0 / 5.8 | 7.3 / 9.7 (hot) |
  | Nested 8x8, free / pinned | 3.12 / 3.26 | 25.4 / 22.9 | 14.6 / 8.1 (hot) |
  | T ≤ 8, overhead | | unchanged within 10% | |

  In these, a waiting thread waits longer than 5 µs although nothing is computing next to it:
  at T=16 the fork must reach 15 workers, and in nested regions each inner team waits for the
  outer one. The waiters fall asleep, and each region then pays wake-ups through the kernel.
- **Spin sweep on regions where nothing computes between waits** (commit 42a3b0b8e, `free`, µs):

  | `POMP_SPIN` | overhead T=8 | overhead T=16 | fork/join T=16 | nested 4x4 | nested 8x8 |
  |---|---|---|---|---|---|
  | 0 | 18.6 | 27.2 | 14.7 | 18.2 | 43.1 |
  | 5 | 1.68 | 6.12 | 7.70 | 3.36 | 24.8 |
  | 20 | 1.75 | 2.14 | 1.13 | 2.06 | 12.3 |
  | 50 | 1.74 | 2.24 | 1.11 | 2.28 | 3.32 |
  | 100 | 1.75 | 2.31 | 1.06 | 2.31 | 3.17 |
  | 200000 | 1.73 | 2.44 | 1.31 | 2.41 | 3.91 |

  Sleeping makes these regions 10–20x slower, and the cost falls off a cliff once the spin
  covers the wait: 20 µs suffices for one level and nested 4x4, nested 8x8 needs 50 µs. Spinning
  0.2 s is slightly worse than 100 µs everywhere (spinning threads slow each other's SMT
  siblings). Each wake-up through the kernel is slow enough that one late region makes the next
  waits longer than the spin, so threads keep falling asleep (a cascade; to be confirmed by the
  counters of the next run).
- **Counters confirm a cascade** (pomp built with `-DPOMP_STATS`, commit c8b0916b8), sleeps and
  posts per region:

  | `POMP_SPIN` | T=8 | T=16 | nested 8x8 |
  |---|---|---|---|
  | 5 | 0 sleeps | 9.1 idle sleeps, 8.8 posts at forks | 6.0 idle + 3.5 team-wait sleeps, 9.1 posts |
  | 20 | 0 | 0 | 1.3 sleeps, 1.3 posts |
  | 100 | 0 | 0 | 0.2 |
  | 200000 | 0 | 0 | 0 |

  Spinning, a T=16 worker waits about 1 µs between regions, far below 5 µs, yet 9 of 15 workers
  sleep in every region. Once some sleep, the leader posts to each (about 0.6 µs each, in turn),
  and the last one runs about 2 µs later: the next wait exceeds 5 µs, so they sleep again. At T=8
  it never starts. A spin of 20 µs breaks it at one level, 100 µs in nested 8x8.
- **A faster wake-up mechanism helps little** (zbench part 8, pinned): one post costs 370–570 ns
  to the poster, and the sleeper runs 1.6–2.4 µs after it, with a mutex and condition (pomp's
  `Post`) as with a futex (5–10% faster). Waking 15 sleepers, the last runs 10.3 µs after the
  first post when posted in turn, 8.8 µs through a 4-ary tree, 8.1 µs with a futex tree. The
  floor is the kernel's 1.6–2.4 µs from wake-up to running, not pomp's code: no mechanism makes
  waking 15 sleepers cheap, so the fix is not to let them fall asleep in short waits.
- **Hence an adaptive spin per thread** (`POMP_SPIN=auto`, the default from commit f99008082).
  A faster wake-up mechanism gains only 15–20% (below), so it was not done. Results, µs:

  | | `auto` | best fixed (value) | 5 µs fixed | 200000 µs fixed |
  |---|---|---|---|---|
  | overhead T=8 / T=16, back to back | 1.78 / 2.30 | 1.60 (5) / 2.19 (20) | 1.60 / 3.85 | 1.81 / 2.51 |
  | nested 4x4 / 8x8 | 2.25 / 3.84 | 1.94 (20) / 3.45 (100) | 2.45 / 23.1 | 2.50 / 3.90 |
  | sleeps per region, T=16 / nested 8x8 | 0 / 0.01 | | 8.3 / 9.8 | 0 / 0 |
  | sibling, 200 µs work, barrier / join,ldr-short (T=2) | 25.6 / 27.8 | 25.9 / 28.6 (0) | 26.7 / 28.6 | 63.8 / 90.0 |
  | sibling, 20 µs work, barrier / join (T=2) | 6.2 / 7.0 | 4.4 / 4.1 (0) | 4.9 / 5.0 | 6.2 / 7.0 |
  | own core, 20 µs work, join,ldr-short (T=2 / T=4) | 1.26 / 1.65 | 1.09 / 1.66 (20) | 3.0 / 3.2 | 1.23 / 1.63 |
  | own core, 200 µs work (T=2 / T=4) | 9.4–14.8 / 16.7–17.1 | within 1–2 µs | | |

  - **It meets the three goals:** back-to-back and nested regions do not sleep (0 and 0.01
    sleeps per region) and are within 5–10% of the best fixed spin; long uneven waits next to a
    sibling sleep and match sleeping at once (25.6 against 25.9 µs); on its own core it is
    within 0.2 µs of the best fixed spin on short waits and within 1–2 µs on long ones.
  - **What it gives up: short waits next to a computing sibling.** With 20 µs of work the wait
    is about 18 µs, under the 100 µs ceiling, so `auto` spins through it, like 200000 (6.2
    against 4.4 µs at `POMP_SPIN=0`). That is by design: it cannot tell a short wait next to a
    computing sibling from a short wait between back-to-back regions, and sleeping in the
    second costs 10x more. A ceiling of 20 µs (`POMP_SPIN=auto:20`) would sleep through these
    waits but brings back the nested 8x8 slowdown (16.6 µs at a fixed 20).
  - **One-level overhead at T ≤ 8 rose 5–15% since the 200000-iteration spin** (T=8 pinned 0.64 →
    0.76 µs, barrier 0.21 → 0.31 µs; also at a fixed 5 µs, so it comes with the timed spin, not
    with `auto`). Likely the clock read every 64 spins, or the clock read at the start of a
    wait; not yet checked.
- **So no single fixed spin time suits both kinds of workload.** Uneven compute next to a sibling
  wants ≤ 5 µs (each extra µs of spin costs about 0.3 µs of the sibling's work: 25 → 28 µs excess
  at 20 µs); back-to-back and nested regions want ≥ 50 µs. A faster wake-up would move the
  second requirement down; an adaptive spin could serve both.
- With each runtime at its best wait policy, pomp and libomp are within 1–2 µs on every row:
  on real work, the wait policy matters more than the fork/join design.
- The 26 µs that remain in `sibling` when sleeping are most likely the short thread's own work
  (about 20 µs at 2500/250) slowing the long one while they share the core, not a runtime cost.

## Things to avoid when measuring

- Comparing runtimes at one placement: each has a different best one; at `pinned` pomp looked up
  to 5.6x (region) and 11x (barrier) faster. Compare each runtime at its own best placement.
- Calling "thread *t* on CPU *t*" just "pinned": it packs two threads per core, so it mixes
  sharing a core with being near. Name the placement, and vary one factor at a time.
- Taking a zbench design for pomp without matching pomp's data layout: zbench F used two lines
  per worker against pomp's one, and overstated pomp's wake loop 3x.
- Trusting Mac results for Z: prefetching and the shared wake word won on the Mac, not on Z, and
  the Mac cannot pin threads.
- Expecting a faster wake-up mechanism to fix sleeping in short waits: on Z the kernel takes
  1.6–2.4 µs to run a woken thread whatever the mechanism (futex, condition, tree).
- Choosing a spin time from one kind of workload: 5 µs was best on uneven compute and 2–8x
  worse on back-to-back and nested regions. Time every change to waits on both.
- Validating a wait-protocol change with a benchmark where nothing sleeps: the release store
  looked free because no thread ever took the sleep path, where it would be wrong.
- Timing `work` while other threads may spin on its SMT sibling: libomp's `pinned` work was 1.45x
  slower for that reason. Make sure waiters are asleep or on other cores.
- Relying on empty-region timings for wait policy: nothing computes while threads spin, so SMT
  interference is invisible. Use uneven-work tests (section 6).
- Calling the time for a wake-up to arrive a latency: it includes the wake loop. Use zbench
  part 2 (ping-pong) for latencies.
- Judging a wait policy by its spin count: `POMP_SPIN` iterations last very different times in
  pool and barrier waits. Express spin limits as time, and check where a thread actually sleeps.
- Reusing `run_z.sh`'s results directory across versions: a stale result file entered the
  best-of summary once. The script now clears it at each run.
- Concluding from one run of libomp `free`: its numbers move 10–15% from run to run.
- Assuming an SMT sibling is closer: on Z it is no closer than another core of the chip (42
  against 36–40 ns).

## Open items

- **Find the 5–15% one-level regression at T ≤ 8 since the timed spin** (section 6): time a
  `SpinBudget` that reads the clock less often (every 256 or 1024 spins), and no clock read at
  the start of a wait that ends within its first spins.
- **Put L's order in pomp:** in `Team::leaderLaunch`, wake the workers on other chips first. pomp
  needs each worker's chip: from sysfs (`physical_package_id`, `book_id`, `drawer_id`) for the
  CPU a worker runs on (`sched_getcpu` at pool entry). Then time the one-level runs again.
- **Placement guidance for onnx-mlir on Z:** confine the process, or each model's threads, to one
  chip when the team fits. Decide whether pomp documents it or the host applies it.
- Time nested shapes at each runtime's best placement (`chip-free`, `cores-close`).
- Time T=32 and 64 (`LIMITS="2 4 8 16 32 64"`), where most workers are on other chips: where L,
  and a wake tree with one leader per chip, should matter most.
- Count the lines that move per region, to explain why another chip costs pomp 0.5 µs at T=2.
- Time a region that starts with the workers asleep: the cost of waking sleepers is not
  measured yet.
- Find why libomp pinned by the test is about 0.4 µs slower than libomp bound by itself or
  confined by `taskset`.
