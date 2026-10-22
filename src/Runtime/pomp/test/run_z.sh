#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build pomp outside onnx-mlir and run its correctness tests and benchmarks,
# for Linux on Z (works on any Linux or macOS with clang and libomp).
#
# Usage, from the pomp directory:
#   test/run_z.sh [build-dir]           # default build-dir: ./build-z
# Environment:
#   CC=clang                            # needs -fopenmp and omp.h
#   OMPFLAGS=-fopenmp                   # compile flags for OpenMP code
#   OMPLIBS=                            # link flags for libomp, if not found
#   LIMITS="4 8 16 32 64"               # thread limits for the benchmarks
#   QUICK=1                             # shorter correctness runs
#   BASE=old/pomp.c                     # also time this older pomp.c, e.g.
#     git show 6e2d4e906:src/Runtime/pomp/pomp.c > /tmp/pomp-before.c
#
# Builds three runtimes and the same tests against each, plus libomp:
#   pomp-xor    pomp as is: team reuse and one XOR wake-up
#   pomp-epoch  -DPOMP_EPOCH_WAKE: team reuse, one epoch per worker
#   pomp-key    -DPOMP_USE_PTHREAD_KEY: as pomp-xor, with a pthread key
#   pomp-before BASE, if given: timed only
#   libomp      the compiler's libomp, for the digests and the reference timings
# Please send back the whole output, with `lscpu` and `lscpu -e`.

set -u
CC=${CC:-clang}
OMPFLAGS=${OMPFLAGS:--fopenmp}
OMPLIBS=${OMPLIBS:-}
B=${1:-build-z}
LIMITS=${LIMITS:-"4 8 16 32 64"}
BASE=${BASE:-}
N=$([ "${QUICK:-0}" = 1 ] && echo 2000 || echo 20000)
mkdir -p "$B"
fail=0

note() { printf '\n== %s\n' "$*"; }
bad() { echo "FAIL: $*"; fail=$((fail + 1)); }

note "build ($($CC --version | head -1))"
OPT="-O2 -std=gnu11"
case $(uname -m) in s390x) OPT="$OPT -march=native" ;; esac
for v in xor epoch key; do
  case $v in
  xor) D= ;;
  epoch) D=-DPOMP_EPOCH_WAKE ;;
  key) D=-DPOMP_USE_PTHREAD_KEY ;;
  esac
  mkdir -p "$B/$v"
  $CC $OPT $D -Wall -c pomp.c -o "$B/$v/pomp.o" &&
    $CC $OPT $OMPFLAGS -DPOMP -c test/pomp_test.c -o "$B/$v/pomp_test.o" &&
    $CC "$B/$v/pomp_test.o" "$B/$v/pomp.o" -pthread -o "$B/$v/pomp_test" &&
    $CC $OPT test/pomp_host_test.c "$B/$v/pomp.o" -pthread -o "$B/$v/pomp_host_test" ||
    { echo "build of pomp-$v failed"; exit 1; }
done
TIMED="xor epoch"
if [ -n "$BASE" ]; then
  mkdir -p "$B/before"
  cp "$BASE" "$B/before/pomp.c" && cp pomp.h "$B/before/" &&
    $CC $OPT -c "$B/before/pomp.c" -o "$B/before/pomp.o" &&
    $CC "$B/xor/pomp_test.o" "$B/before/pomp.o" -pthread \
      -o "$B/before/pomp_test" || { echo "build of BASE failed"; exit 1; }
  TIMED="before $TIMED"
fi
$CC $OPT $OMPFLAGS test/pomp_test.c $OMPLIBS -o "$B/pomp_test_libomp" || exit 1
$CC $OPT $OMPFLAGS -fopenmp-enable-irbuilder -c test/pomp_irbuilder_test.c \
  -o "$B/irb.o" && for v in xor key; do
  $CC "$B/irb.o" "$B/$v/pomp.o" -pthread -o "$B/$v/irb"
done
$CC $OPT -pthread test/pomp_zbench.c -o "$B/pomp_zbench" || exit 1
if command -v objdump >/dev/null && [ "$(uname -m)" = s390x ]; then
  echo "interlocked atomics in pomp.o: $(objdump -d "$B/xor/pomp.o" |
    grep -c -E '\b(laxg|laog|lang|laag)\b')"
fi

note "digests against libomp"
for L in 1 2 3 4 5 8 16; do
  a=$(OMP_THREAD_LIMIT=$L "$B/xor/pomp_test" static)
  b=$(OMP_THREAD_LIMIT=$L "$B/pomp_test_libomp" static 2>/dev/null)
  [ "$a" = "$b" ] || bad "static at limit $L differs from libomp"
done
for E in OMP_DUMMY=1 OMP_NUM_THREADS=4,4 OMP_MAX_ACTIVE_LEVELS=2; do
  for L in 4 8 16; do
    a=$(env $E OMP_THREAD_LIMIT=$L "$B/xor/pomp_test" nested)
    b=$(env $E OMP_THREAD_LIMIT=$L "$B/pomp_test_libomp" nested 2>/dev/null)
    [ "$a" = "$b" ] || bad "nested with $E at limit $L differs from libomp"
  done
done
for v in xor key; do
  [ -x "$B/$v/irb" ] && for L in 1 4 18; do
    OMP_THREAD_LIMIT=$L "$B/$v/irb" | grep -q ok || bad "irbuilder pomp-$v $L"
  done
done
echo "done"

note "stress, barriers and host scenarios"
runs=0
run() {
  runs=$((runs + 1))
  out=$("$@" 2>&1) || { bad "$*"; echo "$out" | tail -5; return; }
  echo "$out" | grep -qE 'bad [1-9]|errors [1-9]' && { bad "$*"; echo "$out"; }
}
for v in xor epoch; do
  T="$B/$v"
  for L in 2 3 4 8 16 17; do
    for S in 0 30 200000; do
      for TB in 2 65; do
        E="env OMP_THREAD_LIMIT=$L POMP_SPIN=$S POMP_TREE_BARRIER=$TB"
        run $E "$T/pomp_test" stress $N
        run $E "$T/pomp_test" barriers $((N / 4))
        run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_test" stress $((N / 4))
        run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_host_test" nest
        run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_host_test" steal $((N / 4))
        for sc in fresh reuse server late overlap errors tworoots; do
          run $E "$T/pomp_host_test" $sc
        done
        [ $L -ge 3 ] && run $E "$T/pomp_host_test" swap
        [ $S = 0 ] && run $E "$T/pomp_host_test" sleepers &&
          run $E "$T/pomp_host_test" sleepjoin
      done
    done
  done
done
echo "$runs runs, $fail failures so far"

note "fork/join, barrier and loop, us (median of 3 runs of 100000)"
med() { sort -n | sed -n 2p; }
field() { sed -n "s/.*$1 \([0-9.]*\).*/\1/p"; }
printf '%-8s %-12s %10s %10s %10s\n' T runtime fork/join barrier loop
for L in $LIMITS; do
  for v in $TIMED libomp; do
    if [ $v = libomp ]; then
      X="$B/pomp_test_libomp"
      E="env KMP_BLOCKTIME=200 OMP_NUM_THREADS=$L OMP_PROC_BIND=close"
    else
      X="$B/$v/pomp_test"
      E=env
    fi
    fj=$(for r in 1 2 3; do $E OMP_THREAD_LIMIT=$L "$X" bench 100000 2>/dev/null | field join; done | med)
    ba=$(for r in 1 2 3; do $E OMP_THREAD_LIMIT=$L "$X" bench 100000 2>/dev/null | field barrier; done | med)
    lo=$(for r in 1 2 3; do $E OMP_THREAD_LIMIT=$L "$X" loop 100000 2>/dev/null | sed -n 's/.*, \([0-9.]*\) us.*/\1/p'; done | med)
    printf '%-8s %-12s %10s %10s %10s\n' $L $v "$fj" "$ba" "$lo"
  done
done

note "pomp_zbench, section 4 (fork/join designs)"
"$B/pomp_zbench" $(echo $LIMITS | tr ' ' ',') 2>&1 | sed -n '/^4\./,/^5\./p'

note "summary: $fail failures"
exit $((fail != 0))
