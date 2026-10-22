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
#   OMPLIBS=                            # link flags for libomp; by default,
#                                       # found next to $CC, or none. May name
#                                       # the archive itself, e.g. onnx-mlir's:
#     OMPLIBS="$PWD/../../../build/Debug/lib/libompruntime.a -ldl"
#   LIMITS="4 8 16 32 64"               # thread limits for the benchmarks
#   QUICK=1                             # shorter correctness runs
#   BASE_REF=origin/pomp-first-attempt  # git ref of pomp before team reuse
#   BASE=old/pomp.c                     # or that pomp.c as a file, if this
#                                       # copy of pomp is not in a git clone
#   BASE_REF= BASE=                     # both empty: skip the "before" pomp
#
# Builds four runtimes and the same tests against each, plus libomp:
#   pomp-xor    pomp as is: team reuse and one XOR wake-up
#   pomp-epoch  -DPOMP_EPOCH_WAKE: team reuse, one epoch per worker
#   pomp-key    -DPOMP_USE_PTHREAD_KEY: as pomp-xor, with a pthread key
#   pomp-before pomp before team reuse (BASE_REF or BASE), tested and timed
#   libomp      the compiler's libomp, for the digests and the reference timings
# Please send back the whole output, with `lscpu` and `lscpu -e`.

set -u
CC=${CC:-clang}
OMPFLAGS=${OMPFLAGS:--fopenmp}
OMPLIBS=${OMPLIBS:-}
B=${1:-build-z}
LIMITS=${LIMITS:-"4 8 16 32 64"}
BASE=${BASE:-}
BASE_REF=${BASE_REF-origin/pomp-first-attempt}
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
# pomp before team reuse: from BASE if given, else from BASE_REF. It has the
# same interface, so it runs today's tests, the new S13 included.
TIMED="xor epoch"
TESTED="xor epoch"
mkdir -p "$B/before"
if [ -n "$BASE" ]; then
  cp "$BASE" "$B/before/pomp.c" || exit 1
  echo "before: $BASE"
elif [ -n "$BASE_REF" ] &&
  git show "$BASE_REF:./pomp.c" >"$B/before/pomp.c" 2>/dev/null; then
  echo "before: $BASE_REF ($(git rev-parse --short "$BASE_REF"))"
else
  rm -f "$B/before/pomp.c"
  echo "before: skipped (no BASE, and BASE_REF '$BASE_REF' not found; try" \
    "git fetch origin)"
fi
if [ -f "$B/before/pomp.c" ]; then
  cp pomp.h "$B/before/" &&
    $CC $OPT -c "$B/before/pomp.c" -o "$B/before/pomp.o" &&
    $CC "$B/xor/pomp_test.o" "$B/before/pomp.o" -pthread \
      -o "$B/before/pomp_test" &&
    $CC $OPT test/pomp_host_test.c "$B/before/pomp.o" -pthread \
      -o "$B/before/pomp_host_test" || { echo "build of before failed"; exit 1; }
  TIMED="before $TIMED"
  TESTED="before $TESTED"
fi
echo "after:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo this copy)" \
  "($(git rev-parse --short HEAD 2>/dev/null || echo no git))" \
  "$(git diff --quiet -- . 2>/dev/null || echo "+ local changes")"
# libomp, for the digests and the reference timings. If it is not on the
# linker's path, look in the lib directories of $CC's installation. Without
# it, only the comparisons with libomp are skipped.
if [ -z "$OMPLIBS" ] &&
  ! echo 'int main(void){return 0;}' |
  $CC $OMPFLAGS -x c - -o "$B/omp_probe" 2>/dev/null; then
  R=$(dirname "$(dirname "$(command -v "$CC")")")
  for d in "$R"/lib "$R"/lib/*-linux-gnu "$R"/lib64 \
    $($CC -print-resource-dir 2>/dev/null)/../..; do
    if ls "$d"/libomp.so* "$d"/libomp.a >/dev/null 2>&1; then
      OMPLIBS="-L$d -Wl,-rpath,$d"
      break
    fi
  done
fi
# With OMPLIBS, link without -fopenmp, which would add -lomp: OMPLIBS may
# then name an archive of another name, such as onnx-mlir's libompruntime.a.
HAVE_LIBOMP=1
$CC $OPT $OMPFLAGS -c test/pomp_test.c -o "$B/pomp_test_libomp.o" &&
  if [ -n "$OMPLIBS" ]; then
    $CC "$B/pomp_test_libomp.o" $OMPLIBS -pthread -lm -o "$B/pomp_test_libomp"
  else
    $CC $OMPFLAGS "$B/pomp_test_libomp.o" -o "$B/pomp_test_libomp"
  fi || HAVE_LIBOMP=0
# An onnx-mlir built with ONNX_MLIR_USE_POMP=ON has pomp as libompruntime.a:
# comparing with it would compare pomp with itself.
if [ $HAVE_LIBOMP = 1 ] && command -v nm >/dev/null &&
  nm "$B/pomp_test_libomp" 2>/dev/null | grep -q pomp_open_pool; then
  echo "OMPLIBS ($OMPLIBS) is pomp, not libomp"
  HAVE_LIBOMP=0
fi
if [ $HAVE_LIBOMP = 1 ]; then
  [ -n "$OMPLIBS" ] && echo "libomp: $OMPLIBS"
else
  echo "libomp not found (set OMPLIBS=-L<dir>, or to the archive):" \
    "skipping the digests and the libomp timings"
fi
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
[ $HAVE_LIBOMP = 1 ] && for v in $TESTED; do
  for L in 1 2 3 4 5 8 16; do
    a=$(OMP_THREAD_LIMIT=$L "$B/$v/pomp_test" static)
    b=$(OMP_THREAD_LIMIT=$L "$B/pomp_test_libomp" static 2>/dev/null)
    [ "$a" = "$b" ] || bad "pomp-$v static at limit $L differs from libomp"
  done
  for E in OMP_DUMMY=1 OMP_NUM_THREADS=4,4 OMP_MAX_ACTIVE_LEVELS=2; do
    for L in 4 8 16; do
      a=$(env $E OMP_THREAD_LIMIT=$L "$B/$v/pomp_test" nested)
      b=$(env $E OMP_THREAD_LIMIT=$L "$B/pomp_test_libomp" nested 2>/dev/null)
      [ "$a" = "$b" ] ||
        bad "pomp-$v nested with $E at limit $L differs from libomp"
    done
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
for v in $TESTED; do
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
  echo "pomp-$v: $runs runs, $fail failures so far"
done

note "fork/join, barrier and loop, us (median of 3 runs of 100000)"
med() { sort -n | sed -n 2p; }
field() { sed -n "s/.*$1 \([0-9.]*\).*/\1/p"; }
printf '%-8s %-12s %10s %10s %10s\n' T runtime fork/join barrier loop
for L in $LIMITS; do
  for v in $TIMED $([ $HAVE_LIBOMP = 1 ] && echo libomp); do
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
