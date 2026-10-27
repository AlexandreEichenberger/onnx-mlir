# Running `pomp_zbench` on Linux on Z

To build pomp itself and time it against libomp on the machine, outside onnx-mlir, use
`test/run_z.sh` instead. From the pomp directory:

```sh
test/run_z.sh 2>&1 | tee pomp-test.log
# if libomp is not on the linker's path, name it, e.g. onnx-mlir's copy:
OMPLIBS="$PWD/../../../build/Debug/lib/libompruntime.a -ldl" test/run_z.sh 2>&1 | tee pomp-test.log
```

It needs a clang with `-fopenmp`, and takes about 20–40 minutes (`QUICK=1` for shorter
correctness runs). It checks the digests against libomp, runs every stress and host scenario at
limits 2–17, then times pomp and libomp (`KMP_BLOCKTIME=200`, hot inner teams) with threads
free and pinned the same way (`CPUS`, default those allowed to the process):

- one level, team sizes `LIMITS` (default 2 4 8 16): empty fork/join, barrier, loop region;
- nested, shapes `NESTS` (default 4x4 4x8 8x4 8x8): an outer team whose members each fork an
  inner team that runs a loop;
- `pomp_zbench` sections 1 and 2, then 4 and 6 with an empty body and with `POMP_ZB_WORK=2000`.

Please send the whole output with `lscpu -e`.

`pomp_zbench.c` is self-contained: C11 and pthreads, with no OpenMP and no other files needed.
It measures the costs that decide pomp's fork/join design on a given machine.

## Build

```sh
gcc   -O2 -std=gnu11 -march=native -pthread pomp_zbench.c -o pomp_zbench
# or
clang -O2 -std=gnu11 -march=native -pthread pomp_zbench.c -o pomp_zbench
```

If `-march=native` is rejected, use `-march=z16` or `-march=z17`, whichever the compiler
knows. Any target from z196 on gives single-instruction atomics.

Check that the atomics compiled to the interlocked-access instructions, not to CS loops:

```sh
objdump -d pomp_zbench | grep -c -E '\b(laxg|laog|lang|laag)\b'   # expect > 0
```

## Run

Please run these four on an otherwise idle LPAR, and send back the whole output of each:

```sh
lscpu; lscpu -e                                   # topology: cores, books, drawers
./pomp_zbench                                     # all CPUs, T = 2,4,8,16,32,64
POMP_ZB_ONE_PER_CORE=1 ./pomp_zbench              # one SMT thread per core
POMP_ZB_WORK=2000 ./pomp_zbench 4,8,16            # regions with a small body
```

If the LPAR has more than 64 CPUs, also try a run restricted to one drawer, for example
`POMP_ZB_CPUS=0-31 ./pomp_zbench`. Choose the range from `lscpu -e`. Each full run takes about
1–3 minutes.

## Environment variables

| Variable | Meaning |
|---|---|
| `POMP_ZB_CPUS` | CPUs to use, in order (`0,2,4-11`); thread *t* is pinned to the *t*-th one. Default: all allowed CPUs |
| `POMP_ZB_ONE_PER_CORE` | `1` keeps only the first SMT thread of each core |
| `POMP_ZB_ITERS` | Regions and barriers per measurement (default 50000) |
| `POMP_ZB_WORK` | Dummy loop iterations each thread runs per region (default 0) |
| `POMP_ZB_SECTIONS` | Sections to run (`4,6`); default 1–6 |

## What each section tells us

| Section | Question it answers |
|---|---|
| 1. Uncontended atomics | Cost of each operation pomp uses: `LAXG`, `LAOG`, `LANG`, `LAAG`, `CSG`, and a seq_cst store or fence (`bcr 14,0`) |
| 2. Cache-line transfer CPU 0 ↔ k | The base cost *c* of moving a line between two CPUs, by distance: same core (SMT), same chip, same drawer, other drawer. This scales every O(T) step |
| 3. Contended `fetch_add` | How badly T threads hitting one line serialize: the cost of a counter join or a central barrier |
| 4. Fork/join designs | Which wake-up and join design to implement (see below) |
| 5. Barrier designs | Whether a tree barrier pays off at these T |
| 6. Anatomy of a region | Where F's region time goes: the wake loop, the workers' wake-up, and the leader's join after the last worker is done; and which worker is last |

Fork/join designs in section 4, all with spinning workers and an empty body unless
`POMP_ZB_WORK` is set:

| | Wake-up | Join | Note |
|---|---|---|---|
| A | one epoch per worker + per-worker assignment | one shared counter | the first pomp |
| B | one XOR on a shared activate word | one shared counter | |
| C | one XOR on a shared activate word | per-worker done flags, polled by the leader | the `pomp-bgq-attempt` fork |
| D | one XOR on a shared activate word | 4-ary tree | |
| E | 4-ary tree | 4-ary tree | close to libomp's default |
| F | one epoch per worker + per-worker assignment | per-worker done flags, polled by the leader | pomp today |
| F' | as F, waking workers T-1 down to 1 | as F | the last woken is polled first: the worst order for F's scan |
| G | as F | the leader sweeps a mask of the workers not yet done, reading each flag until it is done and never again | |
| G' | as G, waking workers T-1 down to 1 | as G | |
| H | as F | done bytes, 8 to a line; the leader reads one word per 8 workers | |
| I | as F, with epoch, assignment and done flag on one line per worker | per-worker done flags, on that line | pomp's layout |
| J | as I, storing the epoch (release) instead of a seq_cst `fetch_add` | as I | |
| K | as J, prefetching every worker's line for store first | as I | |
| L | as J, waking the workers farthest from the leader first | as I | farthest by section 2's latency |

F against F' shows how much the order of F's scan costs; G against F' shows how much the sweep
recovers. I to L attack the wake loop: I against F is the cost of a second line per worker,
J against I that of the serializing `fetch_add`, K against J whether the line transfers can be
started together, and L against J whether starting the farthest ones first helps. Section 6 times F, F', G and G' with a clock every CPU shares (`STCKF` on Z), as ns
after the leader starts waking:

| Column | Meaning |
|---|---|
| fork | the leader's wake loop |
| woken1st, wokenlst | when the first and the last worker see their wake-up |
| donelast | when the last worker is done, before its done store |
| tail | from `donelast` to the end of the leader's join: what the join adds to the critical path |
| region | the whole region, as the leader sees it |
| last w, w=1, w=T-1 | which worker is done last: the mean, and how often it is the first or the last one F polls |
| alone | the wake loop with no worker spinning, its lines in the leader's cache: its instructions alone |

Section 6 first prints the one-way latency from the leader's CPU to each worker's, which L uses.

Set `POMP_ZB_SECTIONS` (for example `4,6`) to run only some sections.

By default, sections 4 and 6 time only E (libomp-like), F (the earlier reference), I (pomp's
layout) and L (the candidate); the other designs are settled (see `../POMP_Z_Tuning.md`). Build
with `-DPOMP_ZB_ALL_DESIGNS` to time them all again. `run_z.sh` skips sections 1 and 2 unless
`ZB_MAP=1`.

## Reference: macOS, 18-core Apple Silicon (not pinned)

Fork/join, µs per region:

| Design | T=2 | 4 | 8 | 16 |
|---|---|---|---|---|
| A | 0.18 | 0.33 | 1.19 | 2.49 |
| B | 0.18 | 0.25 | 0.59 | 0.86 |
| C | 0.16 | 0.19 | 0.54 | 0.61 |
| D | 0.16 | 0.19 | 0.69 | 0.68 |
| E | 0.12 | 0.20 | 0.72 | 0.90 |

On this machine, the XOR activate word (B) alone cuts a 16-thread fork/join from 2.5 to 0.86 µs.
Adding per-worker done flags (C) brings it to 0.61 µs, below libomp's 1.68 µs measured with
`pomp_test bench`. A cache line moves between cores in about 57 ns.
