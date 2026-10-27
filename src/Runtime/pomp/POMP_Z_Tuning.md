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
> Record the pomp commit from the log's `pomp:` line, and whether threads were pinned.

## Setup

- Machine: Linux on Z LPAR, 106 CPUs (0–105). Pinned runs put thread *t* on the *t*-th CPU.
- Tools: `test/run_z.sh` builds pomp from source, checks it against libomp and times both
  runtimes, then runs `test/pomp_zbench`.
- libomp: onnx-mlir's `libompruntime.a`. It is always built as Release, even in an onnx-mlir
  Debug build, so the comparison is fair.
- Times are medians, in µs or ns per region.

## 1. pomp against libomp (runs at pomp f45b331d8, 509ff0353, fbae69c24 and 7d6fe3917)

All runs: 552 stress runs with no failure, and digests that match libomp's. pomp's numbers agree
within about 5% across the runs. The third run (fbae69c24) changes only libomp's free numbers by
more: nested hot teams move 10–15% (8x8 14.4 µs against 12.4–12.9, 8x4 7.3 against 7.9–8.5), and
the T=16 barrier rises from 0.75–0.78 to 0.94 µs. Those libomp free numbers are noisy, and the
ratios that use them (8x8 free 4.6x, T=16 barrier free 1.4x) carry that noise. The fourth run
(7d6fe3917) confirms it: pomp again within 5%, libomp's 8x8 hot free back down to 11.1 µs (3.3x).

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
- libomp gets slower when pinned (T=16: 3.4 µs pinned, 2.6 µs free). Section 6: pinned packs two
  threads per core, free spreads them one per core, and libomp loses more to sharing a core than
  it gains from the shorter distance.
- libomp's `work` column is about 1.5x pomp's when pinned, for the same loop on the same CPUs.
  Overhead subtracts `work`, so the ratios hold. Likely cause (section 6, not yet measured
  directly): libomp's workers keep spinning on the leader's SMT sibling while it computes.
- pomp's pinned overhead jumps from 0.65 to 1.6 µs between T=8 and T=16. Section 5 explains
  it: CPUs 0–11 are one chip, so T=16 is the first size with workers on another chip.

## 2. Fork/join designs (zbench section 4)

The candidate designs are A to F; ZBENCH.md describes them. At T=16, F (pomp today) takes
0.55–0.68 µs and the others 0.69–1.18 µs.

Conclusion: F is the best of A to F, or tied for best, at every T. Unlike on the Mac, a single
shared wake word (C) does not beat a word per worker on Z.

Correction (run fbae69c24): zbench F is not quite pomp. It keeps a worker's assignment on a
second line, where pomp keeps it on the same line as `go` and the done flag. Design I has pomp's
layout and is much faster (section 4), so I, not F, is the reference for pomp today.

## 3. The join's scan of the done flags (zbench designs F', G, G', H; section 6)

The question: the leader polls the done flags in order, 1 to T-1. Does it lose time when an
early-polled worker finishes last?

| Design | What it changes |
|---|---|
| F' | Wakes the workers in reverse, so the worker polled first is woken last: the worst order |
| G, G' | The leader sweeps a mask of the workers not done yet, and reads each flag only until it is done |
| H | Packs the done flags as bytes, 8 to a line |

Section 6 breaks each region into parts with a clock every CPU shares. For F at T=16 with an
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

Section 6 also reports `alone`, the wake loop with no worker spinning: the cost of its
instructions only.

Region time, in µs (zbench section 4), and wake loop at T=16, in ns (section 6):

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

## 5. Machine map and atomics (zbench sections 1 and 2; run fbae69c24)

Uncontended, on one CPU: loads and stores, seq_cst included, take 0.24 ns; a seq_cst fence
0.24 ns; `fetch_add`/`or`/`and`/`xor` 1.27 ns; `exchange` 1.81 ns; a compare-and-swap 1.66 ns.

One-way line transfer from CPU 0 (section 2, every second CPU):

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
- **Pinned, two threads share each core, and that hurts libomp.** Pinned uses CPUs 0, 1, 2, ...
  in order, both SMT threads of each core. libomp is slower pinned than free at T=8 and 16,
  although every worker is closer, and its `work` is 1.45x slower pinned. pomp, which runs few
  instructions per region, gains more from the distance than it loses to the shared core.
- This supports the concern that SMT siblings waste compute, for libomp at least; it is not yet
  measured for pomp's spinning threads next to a thread that computes.
- **The two effects are mixed in "pinned" and "free"**: pinned is near and shares cores; free is
  far and does not. Separating them needs placements that change one at a time (open items).

## Open items

- **Put L's order in pomp:** in `Team::leaderLaunch`, wake the workers on other chips first.
  pomp needs each worker's chip: from sysfs (`physical_package_id` and `book_id`) for pinned
  threads, or from the CPU a worker last ran on (`sched_getcpu`) for free ones. Then measure
  section 1 of `run_z.sh` again.
- Time zbench at T=32 and 64 (`LIMITS="2 4 8 16 32 64"`), where most workers are off-chip:
  that is where L, and a tree with one leader per chip, should matter most.
- Time the sweep (G) with pomp's layout. With I's shorter wake loop, the last worker is
  rarely the one polled last.
- Time a region that starts with the workers asleep: the cost of posting to sleepers is not
  measured yet.
- **Separate distance from SMT sharing** (in `run_z.sh`, next run): T=2 with the worker on the
  leader's SMT sibling, another core of its chip, and another chip; and every T with one thread
  per core (`pinned1`) against two (`pinned`). Both runtimes.
- **Each runtime at its best placement** (in `run_z.sh`, next run): libomp also with its own
  binding, `OMP_PLACES=cores` with `OMP_PROC_BIND=close` and `spread`; a summary takes each
  runtime's best place.
- Explain pomp's 0.38 µs at T=2 on another chip: count the lines that move per region (`go`,
  done, the team's fields, `busy_`).
- Whether spinning threads slow an SMT sibling that computes (zbench section 7, next run): one
  thread computes while a partner on its SMT sibling, its chip or another chip is asleep, spins
  as pomp does, spins and yields as libomp does, or spins with a backoff.
- Since Linux spreads free threads across chips, consider whether pomp should recommend (or
  the host apply) a placement on the leader's chip.
