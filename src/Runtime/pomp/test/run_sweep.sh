#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build pomp outside onnx-mlir and run its correctness tests over a matrix of
# thread limits, barrier kinds and spin settings, at 1 and at 2 active levels.
# No timings and no comparison with libomp: see run_z.sh for those.
#
# Usage, from the pomp directory:
#   test/run_sweep.sh [build-dir]       # default build-dir: ./build-sweep
# Environment:
#   CC=clang                            # needs -fopenmp and omp.h
#   CXX=clang++                         # compiles pomp.cpp; by default, the
#                                       # clang++ next to $CC
#   OMPFLAGS=-fopenmp                   # compile flags for OpenMP code
#   LIMITS="1 2 3 4 5 8 16 17 18 32 64" # thread limits
#   N=10000                             # regions of stress; the other tests
#                                       # run a fraction of it
#
# Builds pomp twice (with __thread, and with -DPOMP_USE_PTHREAD_KEY). For each
# build and limit, runs every test with POMP_TREE_BARRIER 2 (tree barrier from
# 2 threads), 5 (the default) and 65 (central only), and POMP_SPIN 0 (sleep)
# and 200000 (spin). A run fails if it exits nonzero, or prints a nonzero
# `bad` or `errors` count, or `short teams` in nestbench. Limits above the
# CPU count oversubscribe, and are slow with POMP_SPIN=200000.
#
#   1 active level : static (every team size up to the limit), nested (inner
#                    levels serialized), stress, barriers, cycles
#   2 active levels: nested (OMP_MAX_ACTIVE_LEVELS=2 and OMP_NUM_THREADS=4,4),
#                    stress, barriers2, lockstep, nestbench for the shapes of
#                    SHAPES that fit the limit, host nest and steal
#   host protocol  : every pomp_host_test scenario, and nolimit and lateopen,
#                    whose descriptors are allocated while threads run

set -u
CC=${CC:-clang}
CXX=${CXX:-$(dirname "$(command -v "$CC")")/$(basename "$CC" | sed 's/clang/clang++/;s/gcc/g++/')}
OMPFLAGS=${OMPFLAGS:--fopenmp}
B=${1:-build-sweep}
LIMITS=${LIMITS:-"1 2 3 4 5 8 16 17 18 32 64"}
SHAPES=${SHAPES:-"2x2 2x8 4x4 3x5 8x2 4x8 8x8"}
N=${N:-10000}
mkdir -p "$B"

OPT="-O2 -std=gnu11"
CXXOPT="-O2 -std=c++17 -nostdinc++ -fno-exceptions -fno-rtti -fno-threadsafe-statics
  -funwind-tables -Wall -Wextra"
$CC $OPT $OMPFLAGS -DPOMP -c test/pomp_test.c -o "$B/pomp_test.o" || exit 1
for v in tls key; do
  D=$([ $v = key ] && echo -DPOMP_USE_PTHREAD_KEY)
  mkdir -p "$B/$v"
  $CXX $CXXOPT $D -c pomp.cpp -o "$B/$v/pomp.o" &&
    $CC "$B/pomp_test.o" "$B/$v/pomp.o" -pthread -o "$B/$v/pomp_test" &&
    $CC $OPT test/pomp_host_test.c "$B/$v/pomp.o" -pthread \
      -o "$B/$v/pomp_host_test" || { echo "build of pomp-$v failed"; exit 1; }
done

runs=0
fail=0
run() {
  runs=$((runs + 1))
  out=$("$@" 2>&1)
  rc=$?
  if [ $rc != 0 ] ||
    echo "$out" | grep -qE 'bad [1-9]|errors [1-9]|short teams [1-9]'; then
    fail=$((fail + 1))
    echo "FAIL (exit $rc): $*"
    echo "$out" | tail -5
  fi
  [ $((runs % 100)) = 0 ] && echo "progress: $runs runs, $fail failures"
}

for v in tls key; do
  T="$B/$v"
  for L in $LIMITS; do
    for TB in 2 5 65; do
      for S in 0 200000; do
        E="env OMP_THREAD_LIMIT=$L POMP_SPIN=$S POMP_TREE_BARRIER=$TB"
        A2="$E OMP_MAX_ACTIVE_LEVELS=2"
        # 1 active level.
        run $E "$T/pomp_test" static
        run $E OMP_MAX_ACTIVE_LEVELS=1 "$T/pomp_test" nested
        run $E "$T/pomp_test" stress $N
        run $E "$T/pomp_test" barriers $((N / 4))
        run $E "$T/pomp_test" cycles 200
        # 2 active levels.
        run $A2 "$T/pomp_test" nested
        run $E OMP_NUM_THREADS=4,4 "$T/pomp_test" nested
        run $A2 "$T/pomp_test" stress $((N / 4))
        run $A2 "$T/pomp_test" barriers2 $((N / 8))
        [ $L -ge 3 ] && run $A2 "$T/pomp_test" lockstep $((N / 8))
        for sh in $SHAPES; do
          n1=${sh%x*} n2=${sh#*x}
          [ $((n1 * n2)) -le $L ] &&
            run $A2 "$T/pomp_test" nestbench 2000 $n1 $n2
        done
        run $A2 "$T/pomp_host_test" nest
        run $A2 "$T/pomp_host_test" steal $((N / 4))
        # Host protocol. late and swap assume a slot 1 exists.
        for sc in fresh reuse server overlap errors tworoots; do
          run $E "$T/pomp_host_test" $sc
        done
        [ $L -ge 2 ] && run $E "$T/pomp_host_test" late
        [ $L -ge 3 ] && run $E "$T/pomp_host_test" swap
        if [ $S = 0 ]; then
          run $E "$T/pomp_host_test" sleepers
          run $E "$T/pomp_host_test" sleepjoin
          run $E "$T/pomp_host_test" rejoin 50
        fi
      done
    done
    # No OMP_THREAD_LIMIT: the limit, and the descriptors, come from a call.
    run env POMP_SPIN=0 "$T/pomp_host_test" nolimit 50 $L
    run env POMP_SPIN=0 "$T/pomp_host_test" lateopen $L
    run env POMP_SPIN=0 OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_host_test" lateopen $L
  done
done
echo "DONE: $runs runs, $fail failures"
[ $fail = 0 ]
