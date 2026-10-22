# Running `pomp_zbench` on Linux on Z

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

## What each section tells us

| Section | Question it answers |
|---|---|
| 1. Uncontended atomics | Cost of each operation pomp uses: `LAXG`, `LAOG`, `LANG`, `LAAG`, `CSG`, and a seq_cst store or fence (`bcr 14,0`) |
| 2. Cache-line transfer CPU 0 ↔ k | The base cost *c* of moving a line between two CPUs, by distance: same core (SMT), same chip, same drawer, other drawer. This scales every O(T) step |
| 3. Contended `fetch_add` | How badly T threads hitting one line serialize: the cost of a counter join or a central barrier |
| 4. Fork/join designs | Which wake-up and join design to implement (see below) |
| 5. Barrier designs | Whether a tree barrier pays off at these T |

Fork/join designs in section 4, all with spinning workers and an empty body unless
`POMP_ZB_WORK` is set:

| | Wake-up | Join | Note |
|---|---|---|---|
| A | one epoch per worker + per-worker assignment | one shared counter | pomp today |
| B | one XOR on a shared activate word | one shared counter | the proposed cache-hit path |
| C | one XOR on a shared activate word | per-worker done flags, polled by the leader | |
| D | one XOR on a shared activate word | 4-ary tree | |
| E | 4-ary tree | 4-ary tree | close to libomp's default |

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
