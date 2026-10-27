# Tuning pomp on Linux on Z

A record of the experiments run to tune pomp on Z, and what each concluded. Newest results
last. How to run the tools: [test/ZBENCH.md](test/ZBENCH.md).

> **Instructions for updating this file with new measurements.** Please add a summary of any
> new experiments below. When given new output (usually a `run_z.sh` log):
>
> 1. If the log comes from a design that is still open here (see "Open items" and any section
>    marked "not yet run on Z"), fill that section in with the Z results. Replace any Mac-only
>    numbers it uses as a stand-in, and state its conclusions.
> 2. Otherwise, add a new numbered section after the last one. It should say what was asked,
>    what changed (designs or pomp commit), the key numbers in a small table, and the
>    conclusions. Keep it as short as the existing sections.
> 3. If the new data contradicts an earlier conclusion, correct that conclusion where it is
>    stated, and say what changed it.
> 4. Note any drift: if numbers that were already measured (for example, pomp against libomp)
>    move by more than about 5%, record it.
> 5. Update "Open items": remove the ones done and add the new questions.
>
> Record the pomp commit from the log's `pomp:` line, and whether threads were pinned. Define any
> new placement, design or wait policy under "Terms". "Section N" means a section of this file;
> a section of `pomp_zbench`'s output is "zbench part N".

## Setup

- Machine: Linux on Z LPAR, 106 CPUs (0–105). Pinned runs put thread *t* on the *t*-th CPU.
- Tools: `test/run_z.sh` builds pomp from source, checks it against libomp and times both
  runtimes, then runs `test/pomp_zbench`.
- libomp: onnx-mlir's `libompruntime.a`. It is always built as Release, even in an onnx-mlir
  Debug build, so the comparison is fair.
- Times are medians, in µs or ns per region.

## Terms

**Leader** (in OpenMP, the primary thread): the thread that starts a parallel region; it wakes
the other threads of the team, its **workers**, and waits for them at the **join**, the end of
the region. **T**: the number of threads in the team, the leader included.

Machine (section 5): a **core** runs two hardware threads (**SMT**, simultaneous
multithreading); each is a CPU to Linux, and the two are **SMT siblings**. A **chip** here has
6 cores, 12 CPUs (CPUs 0–11 are one chip). A cache line moves between two CPUs of one chip in
about 38 ns, and to any other chip in 72–78 ns.

Placements, the CPUs a team's threads run on:

| Name | Threads pinned? | CPUs |
|---|---|---|
| `free` | no | any; Linux places them |
| `pinned` | yes, thread *t* on one CPU | the *t*-th CPU: 0, 1, 2, ... — two threads per core, filling chip 0 first |
| `pinned1` | yes | one CPU per core: 0, 2, 4, ... — one thread per core; T=8 and 16 reach a second chip |
| `chip-free` | no | the process is limited (`taskset -c 0-11`) to the leader's chip, two CPUs per core; Linux places threads within it |
| `chip1-free` | no | as `chip-free`, with one CPU per core of the chip (`taskset -c 0,2,...,10`) |
| `cores-close` | by libomp | libomp only: `OMP_PLACES=cores OMP_PROC_BIND=close`, one core per thread, packed next to each other |
| `cores-spread` | by libomp | libomp only: `OMP_PLACES=cores OMP_PROC_BIND=spread`, one core per thread, spread over the whole machine |

Fork/join designs in `pomp_zbench` (parts 4 and 6; sections 2 to 4 here). Each is how the leader wakes the workers,
then how it learns that they are done. All workers spin while waiting.

| Design | Wake-up | Join |
|---|---|---|
| A | leader writes an assignment and adds 1 to an epoch word, per worker, each on its own line | each worker adds 1 to one shared counter; the leader waits for T-1 |
| B | leader flips each worker's bit of one shared word (one XOR) | shared counter, as A |
| C | as B | each worker sets its own done flag, on its own line; the leader polls them in order |
| D | as B | 4-ary tree: each thread waits for up to 4 children's done flags, then sets its own |
| E | 4-ary tree: each thread wakes up to 4 children | 4-ary tree, as D (close to libomp's design) |
| F | as A | per-worker done flags, as C (pomp's design, with assignment and epoch on two lines) |
| F' | as F, waking workers T-1 down to 1 | as F |
| G | as F | the leader sweeps the workers not done yet, and reads each done flag only until it is set |
| G' | as F' | as G |
| H | as F | done flags as bytes, 8 per line |
| I | as F, with each worker's epoch, assignment and done flag on one line (pomp's layout) | per-worker done flags, on that line |
| J | as I, writing the epoch with a store instead of an atomic add | as I |
| K | as J, first prefetching every worker's line for writing | as I |
| L | as J, waking the workers on other chips first | as I |

## 1. pomp against libomp (runs at pomp f45b331d8, 509ff0353, fbae69c24, 7d6fe3917, aa1111068)

All runs: 552 stress runs with no failure, and digests that match libomp's. pomp's numbers agree
within about 5% across the runs. The third run (fbae69c24) changes only libomp's free numbers by
more: nested hot teams move 10–15% (8x8 14.4 µs against 12.4–12.9, 8x4 7.3 against 7.9–8.5), and
the T=16 barrier rises from 0.75–0.78 to 0.94 µs. Those libomp free numbers are noisy, and the
ratios that use them (8x8 free 4.6x, T=16 barrier free 1.4x) carry that noise. The fourth run
(7d6fe3917) confirms it: pomp again within 5%, libomp's 8x8 hot free back down to 11.1 µs (3.3x).
The fifth run (aa1111068) agrees with the fourth, within 5% for pomp except T=2 free (0.88 µs,
against 0.77–0.84).

**Correction (run aa1111068): the free and pinned ratios below are not a fair comparison.** Each
runtime does best at a different placement, and libomp does worse when pinned by the test than
when it binds itself. Section 7 compares each runtime at its own best placement: pomp is 1.5–2.5x
faster on overhead and 1.1–3.0x on a barrier. Quote those.

Overhead of a region with one static loop, as the ratio libomp / pomp:

| | T=2 | 4 | 8 | 16 | nested 4x4 to 8x8 |
|---|---|---|---|---|---|
| free | 1.6–1.7x | 1.3–1.4x | 1.3–1.4x | 1.1–1.2x | 1.7–4.0x |
| pinned | 5.5x | 3.4x | 4.1x | 2.0–2.1x | 2.5–4.8x |

The nested ratios are against libomp's hot teams. libomp by default builds every inner team
again in each region, and pomp is 10–35x faster than that.

On a single barrier, pomp is 1.1–3x faster free and 2.8–11x faster pinned. The one exception
is T=4 free, at 0.8–0.9x.

Conclusions:
- pomp is faster than libomp at every size, except a free barrier at T=4. That point looks
  like scheduler noise, and libomp shows it as well.
- The pinned T=2 ratios (5.5x) are the least representative. CPUs 0 and 1 are SMT siblings
  (section 5), but a line moves between them no faster than between two cores of the same chip
  (42 against 36–40 ns), so it is not a cheaper transfer that helps pomp there. Section 6
  explains pomp's 3x gap between pinned and free: free, Linux puts every worker on another chip,
  without migrating it. The free numbers are still the safer ones to quote.
- libomp gets slower when pinned (T=16: 3.4 µs pinned, 2.6 µs free). Section 7: a worker on the
  leader's SMT sibling costs libomp 0.34 µs at T=2, and libomp is also about 0.4 µs slower when
  the test pins its threads than when it binds them itself (cause open).
- libomp's `work` column is about 1.5x pomp's when pinned, for the same loop on the same CPUs.
  Overhead subtracts `work`, so the ratios hold. Cause, confirmed by section 7: libomp's workers
  keep spinning, and yielding, on the leader's SMT sibling while it computes, which slows it
  1.2–1.4x. pomp's workers have gone to sleep by then.
- pomp's pinned overhead jumps from 0.65 to 1.6 µs between T=8 and T=16. Section 5 explains
  it: CPUs 0–11 are one chip, so T=16 is the first size with workers on another chip.

## 2. Fork/join designs (zbench part 4)

The candidate designs are A to F; ZBENCH.md describes them. At T=16, F (pomp today) takes
0.55–0.68 µs and the others 0.69–1.18 µs.

Conclusion: F is the best of A to F, or tied for best, at every T. Unlike on the Mac, a single
shared wake word (C) does not beat a word per worker on Z.

Correction (run fbae69c24): zbench F is not quite pomp. It keeps a worker's assignment on a
second line, where pomp keeps it on the same line as `go` and the done flag. Design I has pomp's
layout and is much faster (section 4), so I, not F, is the reference for pomp today.

## 3. The join's scan of the done flags (zbench designs F', G, G', H; zbench part 6)

The question: the leader polls the done flags in order, 1 to T-1. Does it lose time when an
early-polled worker finishes last?

| Design | What it changes |
|---|---|
| F' | Wakes the workers in reverse, so the worker polled first is woken last: the worst order |
| G, G' | The leader sweeps a mask of the workers not done yet, and reads each flag only until it is done |
| H | Packs the done flags as bytes, 8 to a line |

zbench part 6 breaks each region into pieces with a clock every CPU shares. For F at T=16 with an
empty body, out of 684 ns: the wake loop takes 435 ns, the wake-up reaching the last worker
80 ns, the last worker's run 67 ns, and the join after it (the tail) 100 ns. With pomp's
one-line layout (design I, section 4), the wake loop is only 139 of 430 ns.

Conclusions:
- **The order of the scan barely matters.** The worker polled last is also the last one done
  99% of the time, because the wake loop spreads the workers out more than their work does.
  This holds even with a body of 2000 iterations.
  Caveat (run fbae69c24): that is for F's slow wake loop. With pomp's layout (I, J) the wake
  loop is 3x shorter, and at T=16 the worker polled last is the last one done only 7–10% of
  the time with an empty body (about 50% with a body of 2000). The join tail is still no longer
  than F's (107 against 127 ns), so there is no sign yet that the scan costs, but the sweep has
  not been timed with pomp's layout.
- The worst order (F') costs only 15–30 ns more. The sweep (G') wins most of that back, but G
  gives nothing in F's own order. **The sweep is not worth adding**, unless loads with uneven
  work show otherwise.
- **Packing done flags 8 to a line (H) is worse** (T=16: 0.86 against 0.68 µs). The workers
  contend for the shared line. Dropped.
- The tail at T=16 is mostly one transfer from a far CPU. That distance, not the scan,
  explains the step in overhead from T=8 to T=16. Correction (run fbae69c24): the 136 and 62 ns
  first given here were the times for a wake-up to arrive, not latencies. The measured one-way
  latencies are 36–39 ns to workers 1–11 (same chip) and 75–78 ns to workers 12–15
  (section 5).

## 4. The wake loop (zbench designs I to L; run fbae69c24, pinned)

The question: can the leader's wake loop, the largest part of F's region, be made cheaper? It
does not post to workers that are spinning (they sleep only after `POMP_SPIN` = 200000 spins),
so its cost is cache-line transfers.

| Design | What it changes, over F |
|---|---|
| I | The epoch, the assignment and the done flag share one line per worker, as in pomp. zbench F used two lines. |
| J | I, with a release store of `go` instead of a seq_cst `fetch_add` |
| K | J, prefetching every worker's line for store before the stores |
| L | J, waking the farthest workers first (by measured latency) |

zbench part 6 also reports `alone`, the wake loop with no worker spinning: the cost of its
instructions only.

Region time, in µs (zbench part 4), and wake loop at T=16, in ns (zbench part 6):

| | T=4 | 8 | 16 | 16, body 2000 | wake loop, T=16 |
|---|---|---|---|---|---|
| F | 0.157 | 0.243 | 0.680 | 4.282 | 467 |
| I | 0.097 | 0.160 | 0.413 | 4.035 | 139 |
| J | 0.089 | 0.158 | 0.412 | 4.012 | 134 |
| K | 0.100 | 0.161 | 0.427 | 4.055 | 204 |
| L | 0.089 | 0.160 | **0.300** | **3.872** | 143 |

`alone` is 3–24 ns, so the wake loop's cost is almost all line transfers.

Conclusions:
- **I against F: zbench F overstated pomp's wake loop 3x** (467 against 139 ns at T=16) and
  its region by 40%. pomp already has I's layout, so nothing to change in pomp; zbench's
  reference is now I.
- **J against I: the `fetch_add` costs nothing.** Section 5 agrees: an interlocked add is
  1.3 ns uncontended. Keep `fetchAdd` in pomp.
- **K against J: prefetching does not help on Z**, and costs 70 ns at T=16. On the Mac it
  halved the wake loop. Dropped.
- **L against J: waking the far workers first cuts the T=16 region by 27%** (0.412 to
  0.300 µs) and saves 140 ns with a body of 2000. The four workers on another chip (12–15) get
  their wake-up first, so their 75 ns transfers overlap the near ones instead of coming last.
  At T up to 8 every worker is on the leader's chip and L equals J. **This is the one change
  worth making in pomp.**
  Run 7d6fe3917 confirms it: T=16 region 0.293 against 0.382 µs for I, and 3.85 against
  4.04 µs with a body of 2000.
- With pomp's layout, the done flag sits on the line the leader just wrote, so the join tail
  at T=4 and 8 is about 75 ns, against 45–53 ns for F. The region is still shorter overall.

## 5. Machine map and atomics (zbench parts 1 and 2; run fbae69c24)

zbench part 1, uncontended, on one CPU: loads and stores, seq_cst included, take 0.24 ns; a seq_cst fence
0.24 ns; `fetch_add`/`or`/`and`/`xor` 1.27 ns; `exchange` 1.81 ns; a compare-and-swap 1.66 ns.

One-way line transfer from CPU 0 (zbench part 2, every second CPU):

| CPUs | Where | One-way ns |
|---|---|---|
| 1 | SMT sibling (core 0) | 42 |
| 2–11 | other cores of the same chip (cores 1–5) | 36–40 |
| 12–105 | other chips, every book of the drawer | 72–78 |

Conclusions:
- The cost of an atomic itself is negligible on Z; what counts is how many lines move, and how
  far.
- **There are only two distances: the same chip (about 38 ns) and anywhere else (72–78 ns),**
  across books. All CPUs are in one drawer. A chip here is 6 cores, 12 CPUs.
- An SMT sibling is no closer than another core of the chip.
- A wake order or tree that follows the topology needs only one level: the leader's chip, and
  the rest.

## 6. Where threads run, free and pinned (`pomp_test place`; run 7d6fe3917)

The question: why does free placement cost pomp so much more than libomp (T=2 overhead: pomp
0.27 µs pinned against 0.84 free; libomp 1.47 against 1.38)? The test times empty fork/join
regions one by one, then records each thread's CPU in every region.

Empty fork/join, p50 µs per region, and where the workers were:

| T | pomp pinned | pomp free | libomp pinned | libomp free | pinned: workers | free: workers |
|---|---|---|---|---|---|---|
| 2 | 0.140 | 0.515 | 0.808 | 0.822 | SMT sibling | 100% other chip |
| 4 | 0.223 | 0.661 | 0.865 | 1.032 | same chip | 100% other chip |
| 8 | 0.309 | 0.861 | 1.397 | 1.198 | same chip | 86% other chip |
| 16 | 0.750 | 1.060 | 1.581 | 1.509 | 27% other chip | 81–89% other chip |

In every row: no migration (at most 0.01 per 1000 regions), almost no context switches, all CPUs
horizontally polarized, and no outliers, except libomp free at T=16 (16 involuntary switches per
1000 regions, 1.7% of regions over twice the median, likely its `sched_yield`).

Conclusions:
- **Free placement is not noisy, it is far.** Linux spreads the team one thread per core, and
  onto other chips: at T=2 the worker is on another chip in 100% of regions. It does not move it
  and does not preempt it. The cost is steady (p99 within 20% of p50).
- **pomp pays for the distance, libomp barely does.** pomp's time is mostly line transfers, so
  doubling their distance (38 to 75 ns) costs it 0.38 µs at T=2. That is more than the
  difference in latency times the transfers we count per region; the cause of the rest is open.
  Correction (run aa1111068, section 7): libomp pays for it just as much. With both threads
  pinned, a worker on another chip instead of the leader's chip adds 0.51 µs to pomp and
  0.47 µs to libomp at T=2. libomp seemed not to pay because its pinned runs put the worker on
  the leader's SMT sibling, which costs it about as much.
- **Pinned, two threads share each core, and that hurts libomp.** Pinned uses CPUs 0, 1, 2, ...
  in order, both SMT threads of each core. libomp is slower pinned than free at T=8 and 16,
  although every worker is closer, and its `work` is 1.45x slower pinned. pomp, which runs few
  instructions per region, gains more from the distance than it loses to the shared core.
- This supports the concern that SMT siblings waste compute, for libomp at least. Section 7
  measures it directly, for both: a spinning sibling slows a computing thread about 1.3x.
- **The two effects are mixed in "pinned" and "free"**: pinned is near and shares cores; free is
  far and does not. Separating them needs placements that change one at a time (open items).

## 7. Each runtime at its best placement, and SMT interference (run aa1111068)

The questions: does distance or sharing a core explain the cost of placement; what does each
runtime achieve at its own best placement; and does a waiting thread slow a computing SMT
sibling? New placements, one level: `pinned1`, `chip-free`, `chip1-free`, `cores-close` and
`cores-spread` (defined under Terms).

T=2, both threads pinned, the worker by distance from the leader (overhead, µs):

| Worker on | pomp | libomp |
|---|---|---|
| SMT sibling (CPU 1) | 0.273 | 1.467 |
| another core of the chip (CPU 2) | 0.325 | 1.123 |
| another chip (CPU 12) | 0.837 | 1.590 |

Overhead in µs (lower is better), the best of each runtime's placements:

| T | libomp best | its place | pomp best | its place | ratio | pomp chip-free | libomp free |
|---|---|---|---|---|---|---|---|
| 2 | 0.672 | chip1-free | 0.267 | pinned | 2.5x | 0.319 | 1.373 |
| 4 | 0.832 | chip-free | 0.470 | pinned | 1.8x | 0.474 | 1.837 |
| 8 | 1.246 | chip-free | 0.647 | pinned | 1.9x | 0.734 | 2.445 |
| 16 | 2.337 | cores-close | 1.541 | pinned | 1.5x | — | 2.698 |

Best barrier, libomp against pomp: 0.15/0.05 (3.0x), 0.17/0.16 (1.1x), 0.35/0.21 (1.7x),
0.66/0.42 (1.6x).

zbench part 7: one thread computes on CPU 0 while a partner waits. Time against the
thread alone:

| Partner on | asleep | spin (pomp) | spin + yield (libomp) | spin + backoff |
|---|---|---|---|---|
| SMT sibling, fp loop | 1.00x | 1.29x | 1.40x | 1.21x |
| SMT sibling, integer chain | 1.00x | 1.30x | 1.21x | 1.16x |
| another core or chip | 1.00x | 1.00–1.02x | 1.00–1.02x | 1.00–1.02x |

Conclusions:
- **Distance dominates, for both runtimes.** Another chip costs about 0.5 µs at T=2 for pomp and
  libomp alike. Within the chip, pomp barely cares whether the worker shares the leader's core
  (0.27 against 0.33 µs). libomp does: its sibling is 0.34 µs slower than another core.
- **Keeping the team on the leader's chip recovers almost all of it, without pinning.** pomp
  `chip-free` is within 0–20% of pinned, and libomp's best placement is `chip-free` or
  `chip1-free`, twice as fast as free at T=2 to 8. So limiting the process to one chip
  (`taskset`, or a cgroup's `cpuset`) is enough for a model that fits on a chip. Linux on its
  own spreads the threads over chips.
- **The fair headline: pomp is 1.5–2.5x faster than libomp on a region, and 1.1–3.0x on a
  barrier, each at its own best placement.** The earlier pinned ratios (up to 5.6x and 11x)
  compared libomp at a bad placement.
- **libomp's own binding:** `cores-close` is the best of libomp's own options, and its best at
  T=16. `cores-spread` is as bad as free, because it puts threads on other chips on purpose. At
  T ≤ 8, `close` is a little slower than `chip-free` (T=8: 1.62 against 1.25 µs). It started
  the team at CPU 14 (`OMP_DISPLAY_AFFINITY`), not on CPU 0's chip.
- **Open:** libomp pinned by the test on CPUs 0 and 2 (`pinned1`, and the distance table) is
  1.10–1.12 µs. Bound by itself or confined by `taskset` on the same kind of CPUs, it is
  0.67–0.71 µs. pomp does not show that gap.
- **SMT interference is real: a waiting thread that spins slows its SMT sibling by about 30%.**
  It costs nothing on any other core, and nothing when it sleeps. Yielding does not help; a
  backoff between polls wins back only a third of it. This is the waste in the original
  concern. It applies to pomp too whenever a waiting thread shares a core with one that
  computes: a worker that finishes first and spins in the barrier, or the leader's sibling.
- pomp's empty-region numbers miss this, since nothing computes while its threads spin.

## 8. Does spinning cost more compute than it saves? (not yet run on Z)

The concern: a thread that waits at a barrier or a join can spin, which saves about 1 µs of
wake-up, or sleep. Section 7 found that a spinning thread slows its SMT sibling by about 30%.
If the sibling is still computing, spinning could cost far more work than it saves in
overhead. The test (`pomp_test imbal`, run by `run_z.sh`) measures that tradeoff with uneven
work.

Setup:
- Each region gives even tids LONG passes of work and odd tids SHORT ones. One pass scales 1024
  floats, the loop of the one-level timings, about 0.08 µs. Two sizes: 250/25 passes (about 20 µs
  of long work) and 2500/250 (about 200 µs).
- Sync: `join`, one region per phase, so the short thread waits at the join; `join,ldr-short`,
  the same with the leader short, so the leader waits next to a long worker; `barrier`, 10
  barriers per region, each after a phase of work.
- Placement: `sibling`, tids 2k and 2k+1 on the two CPUs of one core, so every long thread
  computes next to a waiting one; `own-core`, every thread on a core of its own, as a control.
  T=2 and 4, all on the leader's chip.
- Wait policies: pomp `POMP_SPIN` = spins before a waiting thread sleeps (200000 by default,
  then 20000, 2000, and 0 for sleeping at once); libomp `KMP_BLOCKTIME` = time spent spinning
  before sleeping (200 ms as in the other runs, 20 µs, and 0).
- Output: the time per phase, and `excess` = phase − `alone`, where `alone` is the long work
  run by one thread. The excess is the runtime's overhead plus the slowdown of the long work by
  its waiting sibling.

What would settle it: if, in `sibling`, a policy that sleeps sooner (smaller `POMP_SPIN`) has a
smaller excess than the default, then spinning costs more compute than it saves, and pomp
should sleep sooner, or not spin at all next to a computing sibling. `own-core` shows the cost
of sleeping when there is no sibling to slow down.

## Open items

- **Put L's order in pomp:** in `Team::leaderLaunch`, wake the workers on other chips first.
  pomp needs each worker's chip: from sysfs (`physical_package_id` and `book_id`) for pinned
  threads, or from the CPU a worker last ran on (`sched_getcpu`) for free ones. Then measure
  the one-level timings of `run_z.sh` again.
- Time zbench at T=32 and 64 (`LIMITS="2 4 8 16 32 64"`), where most workers are off-chip:
  that is where L, and a tree with one leader per chip, should matter most.
- Time the sweep (G) with pomp's layout. With I's shorter wake loop, the last worker is
  rarely the one polled last.
- Time a region that starts with the workers asleep: the cost of posting to sleepers is not
  measured yet.
- Explain pomp's 0.38 µs at T=2 on another chip: count the lines that move per region (`go`,
  done, the team's fields, `busy_`).
- **Placement guidance for onnx-mlir on Z:** confine the process (or each model's threads) to one
  chip when the team fits, as `taskset` does. Decide whether pomp documents it, or the host
  applies it.
- **Run section 8 on Z.** It times SMT interference with uneven work at joins and barriers,
  for both runtimes, at several wait policies. Then decide pomp's wait policy: spin for how
  long, back off, or sleep at once. A longer backoff and a pause instruction are not timed yet.
- Run the nested shapes with `chip-free` and `cores-close`, to compare nested regions at each
  runtime's best placement too.
- Find why libomp pinned by the test (`pthread_setaffinity_np` per thread) is about 0.4 µs
  slower than libomp bound by itself or confined by `taskset`.
