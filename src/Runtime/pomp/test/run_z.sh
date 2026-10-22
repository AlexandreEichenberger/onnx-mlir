#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build pomp outside onnx-mlir, run its correctness tests, and time it against
# libomp. Written for Linux on Z; works on any Linux or macOS with clang and
# libomp.
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
#   LIMITS="2 4 8 16"                   # team sizes timed
#   NESTS="4x4 4x8 8x4 8x8"             # nested team shapes timed
#   CPUS=                               # CPUs to pin to (default: those this
#                                       # process may use); see "Placement"
#   QUICK=1                             # shorter correctness runs
#
# Builds pomp twice (with __thread, and with -DPOMP_USE_PTHREAD_KEY), runs the
# tests against each, and times pomp against libomp.
#
# Placement: each timing is taken twice, with threads left to the OS
# ("free") and with thread t pinned to the t-th CPU of CPUS ("pinned"), the
# same way for pomp and libomp (POMP_TEST_CPUS in pomp_test). On z/OS, pomp
# cannot choose where the host's threads run, so "free" is the realistic
# case, and "pinned" shows what placement alone is worth.
#
# Please send back the whole output, with `lscpu -e`.

set -u
CC=${CC:-clang}
OMPFLAGS=${OMPFLAGS:--fopenmp}
OMPLIBS=${OMPLIBS:-}
B=${1:-build-z}
LIMITS=${LIMITS:-"2 4 8 16"}
NESTS=${NESTS:-"4x4 4x8 8x4 8x8"}
N=$([ "${QUICK:-0}" = 1 ] && echo 2000 || echo 20000)
mkdir -p "$B"
fail=0

note() { printf '\n== %s\n' "$*"; }
bad() { echo "FAIL: $*"; fail=$((fail + 1)); }

note "build ($($CC --version | head -1))"
echo "pomp: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo this copy)" \
  "($(git rev-parse --short HEAD 2>/dev/null || echo no git))" \
  "$(git diff --quiet HEAD -- . 2>/dev/null || echo "+ local changes")"
OPT="-O2 -std=gnu11"
case $(uname -m) in s390x) OPT="$OPT -march=native" ;; esac
$CC $OPT $OMPFLAGS -DPOMP -c test/pomp_test.c -o "$B/pomp_test.o" || exit 1
for v in tls key; do
  D=$([ $v = key ] && echo -DPOMP_USE_PTHREAD_KEY)
  mkdir -p "$B/$v"
  $CC $OPT $D -Wall -c pomp.c -o "$B/$v/pomp.o" &&
    $CC "$B/pomp_test.o" "$B/$v/pomp.o" -pthread -o "$B/$v/pomp_test" &&
    $CC $OPT test/pomp_host_test.c "$B/$v/pomp.o" -pthread \
      -o "$B/$v/pomp_host_test" || { echo "build of pomp-$v failed"; exit 1; }
done

# libomp, for the digests and the timings. If it is not on the linker's path,
# look in the lib directories of $CC's installation. Without it, only the
# comparisons with libomp are skipped.
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
  -o "$B/irb.o" && for v in tls key; do
  $CC "$B/irb.o" "$B/$v/pomp.o" -pthread -o "$B/$v/irb"
done
$CC $OPT -pthread test/pomp_zbench.c -o "$B/pomp_zbench" || exit 1
if command -v objdump >/dev/null && [ "$(uname -m)" = s390x ]; then
  echo "interlocked atomics in pomp.o: $(objdump -d "$B/tls/pomp.o" |
    grep -c -E '\b(laxg|laog|lang|laag)\b')"
fi

# CPUs for the pinned runs: CPUS, or those this process may use.
if [ -z "${CPUS:-}" ] && command -v taskset >/dev/null; then
  CPUS=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')
fi
CPUS=${CPUS:-0-63}
echo "pinned runs: thread t on the t-th CPU of $CPUS"

note "digests against libomp"
[ $HAVE_LIBOMP = 1 ] && for v in tls key; do
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
for v in tls key; do
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
T="$B/tls"
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
echo "$runs runs, $fail failures"

# Timings: median of 5 runs. libomp spins 200 ms before sleeping, as pomp
# spins POMP_SPIN iterations.
med() { sort -n | sed -n 3p; }
field() { sed -n "s/.*$1 \([0-9.]*\) us.*/\1/p"; }
time5() { # field cmd...
  local f=$1 r
  shift
  for r in 1 2 3 4 5; do "$@" 2>/dev/null | field "$f"; done | med
}
RUNTIMES="pomp $([ $HAVE_LIBOMP = 1 ] && echo libomp)"
exe() { [ $1 = pomp ] && echo "$B/tls/pomp_test" || echo "$B/pomp_test_libomp"; }
LIBENV="KMP_BLOCKTIME=200"

note "one level, us per region (median of 5 runs of 100000)"
printf '%-4s %-7s %-7s %10s %10s %10s\n' T runtime place fork/join barrier loop
for L in $LIMITS; do
  for rt in $RUNTIMES; do
    for place in free pinned; do
      P=$([ $place = pinned ] && echo "POMP_TEST_CPUS=$CPUS")
      E="env $LIBENV OMP_THREAD_LIMIT=$L OMP_NUM_THREADS=$L $P"
      X=$(exe $rt)
      fj=$(time5 join $E "$X" bench 100000)
      ba=$(time5 barrier $E "$X" bench 100000)
      lo=$(time5 region $E "$X" loop 100000)
      printf '%-4s %-7s %-7s %10s %10s %10s\n' $L $rt $place "$fj" "$ba" "$lo"
    done
  done
done

note "nested, outer x inner threads, us per outer region (median of 5 runs of 50000)"
echo "Each outer member forks an inner team that runs one loop and its barrier."
echo "libomp runs twice: by default, which creates every inner team again, and"
echo "with hot inner teams, which it keeps as pomp does. Hot teams get full inner"
echo "teams only without OMP_THREAD_LIMIT (else libomp gives them 1 thread), so"
echo "libomp runs without it. Any short team is flagged: that timing is invalid."
printf '%-6s %-12s %-7s %10s\n' shape runtime place region
short=0
for s in $NESTS; do
  n1=${s%x*}
  n2=${s#*x}
  for rt in $RUNTIMES $([ $HAVE_LIBOMP = 1 ] && echo libomp-hot); do
    case $rt in
    pomp) RE="OMP_THREAD_LIMIT=$((n1 * n2))" ;;
    libomp) RE= ;;
    libomp-hot) RE="KMP_HOT_TEAMS_MAX_LEVEL=2 KMP_HOT_TEAMS_MODE=1" ;;
    esac
    X=$(exe ${rt%-hot})
    for place in free pinned; do
      P=$([ $place = pinned ] && echo "POMP_TEST_CPUS=$CPUS")
      E="env $LIBENV OMP_MAX_ACTIVE_LEVELS=2 $RE $P"
      v=$(for r in 1 2 3 4 5; do $E "$X" nestbench 50000 $n1 $n2 2>/dev/null; done)
      r=$(echo "$v" | field region | med)
      flag=
      if echo "$v" | grep -qE 'short teams ([1-9][0-9]* outer|[0-9]+ outer [1-9])'; then
        flag="  SHORT TEAMS: invalid"
        short=$((short + 1))
      fi
      printf '%-6s %-12s %-7s %10s%s\n' $s $rt $place "$r" "$flag"
    done
  done
done
[ $short = 0 ] || echo "$short nested timings had short teams and are invalid"

note "pomp_zbench, section 4 (fork/join designs; F is pomp today)"
"$B/pomp_zbench" $(echo $LIMITS | tr ' ' ',') 2>&1 | sed -n '/^4\./,/^5\./p'

note "summary: $fail failures"
exit $((fail != 0))
