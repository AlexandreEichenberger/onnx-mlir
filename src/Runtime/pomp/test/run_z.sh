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
#   CXX=clang++                         # compiles pomp.cpp; by default, the
#                                       # clang++ next to $CC
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
CXX=${CXX:-$(dirname "$(command -v "$CC")")/$(basename "$CC" | sed 's/clang/clang++/;s/gcc/g++/')}
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

note "build ($($CC --version | head -1); C++: $CXX)"
echo "pomp: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo this copy)" \
  "($(git rev-parse --short HEAD 2>/dev/null || echo no git))" \
  "$(git diff --quiet HEAD -- . 2>/dev/null || echo "+ local changes")"
OPT="-O2 -std=gnu11"
# pomp.cpp needs no C++ header nor C++ runtime: -nostdinc++ enforces the
# first, and linking every program with $CC (the C driver) the second.
CXXOPT="-O2 -std=c++17 -nostdinc++ -fno-exceptions -fno-rtti -fno-threadsafe-statics
  -funwind-tables"
case $(uname -m) in s390x)
  OPT="$OPT -march=native"
  CXXOPT="$CXXOPT -march=native"
  ;;
esac
$CC $OPT $OMPFLAGS -DPOMP -c test/pomp_test.c -o "$B/pomp_test.o" || exit 1
for v in tls key; do
  D=$([ $v = key ] && echo -DPOMP_USE_PTHREAD_KEY)
  mkdir -p "$B/$v"
  $CXX $CXXOPT $D -Wall -c pomp.cpp -o "$B/$v/pomp.o" &&
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
      run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_test" barriers2 $((N / 8))
      [ $L -ge 3 ] && run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_test" lockstep $((N / 8))
      run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_host_test" nest
      run $E OMP_MAX_ACTIVE_LEVELS=2 "$T/pomp_host_test" steal $((N / 4))
      for sc in fresh reuse server late overlap errors tworoots; do
        run $E "$T/pomp_host_test" $sc
      done
      [ $L -ge 3 ] && run $E "$T/pomp_host_test" swap
      [ $S = 0 ] && run $E "$T/pomp_host_test" sleepers &&
        run $E "$T/pomp_host_test" sleepjoin &&
        run $E "$T/pomp_host_test" rejoin 50
    done
  done
  # No OMP_THREAD_LIMIT: the limit, and the descriptors, come from a call.
  run env POMP_SPIN=0 "$T/pomp_host_test" nolimit 50 $L
  run env POMP_SPIN=0 "$T/pomp_host_test" lateopen $L
done
echo "$runs runs, $fail failures"

# Timings: median of 5 runs. libomp spins 200 ms before sleeping, as pomp
# spins POMP_SPIN iterations. Every timing starts after 1000 untimed regions
# of the same shape, so that libomp's threads exist and are hot: no timing
# includes the creation of a thread.
#
# The two metrics, in us:
#   overhead  the runtime's cost of a parallel region with one static loop
#             (fork, loop init, the loop's barrier, join): the region's time
#             minus the time of its work, one thread's share of the loop run
#             alone;
#   barrier   one barrier inside a running region.
med() { sort -n | sed -n 3p; }
field() { sed -n "s/.*$1 \([-0-9.]*\) us.*/\1/p"; }
time5() { # field cmd...
  local f=$1 r
  shift
  for r in 1 2 3 4 5; do "$@" 2>/dev/null | field "$f"; done | med
}
# 5 runs of a command whose output has several fields: the median of each.
runs5() { # cmd...
  local r
  for r in 1 2 3 4 5; do "$@" 2>/dev/null; done
}
med_of() { field "$1" | med; } # field, on the output of runs5
ratio() { awk -v a="$1" -v b="$2" 'BEGIN { if (a > 0 && b > 0) printf "%.1fx", b / a; else print "-" }'; }
POMPS=pomp
RUNTIMES="$POMPS $([ $HAVE_LIBOMP = 1 ] && echo libomp)"
exe() { case $1 in pomp*) echo "$B/tls/pomp_test" ;; *) echo "$B/pomp_test_libomp" ;; esac; }
LIBENV="KMP_BLOCKTIME=200"
R="$B/results"
mkdir -p "$R"

note "one level, us per region (median of 5 runs of 100000)"
echo "overhead = region - work; work = one thread's share of the loop alone."
printf '%-4s %-10s %-7s %9s %9s | %9s %9s %9s\n' T runtime place \
  overhead barrier region work fork/join
for L in $LIMITS; do
  for rt in $RUNTIMES; do
    for place in free pinned; do
      P=$([ $place = pinned ] && echo "POMP_TEST_CPUS=$CPUS")
      E="env $LIBENV OMP_THREAD_LIMIT=$L OMP_NUM_THREADS=$L $P"
      X=$(exe $rt)
      b=$(runs5 $E "$X" bench 100000)
      lo=$(runs5 $E "$X" loop 100000)
      fj=$(echo "$b" | med_of join)
      ba=$(echo "$b" | med_of barrier)
      re=$(echo "$lo" | med_of region)
      wo=$(echo "$lo" | med_of work)
      ov=$(echo "$lo" | med_of overhead)
      printf '%-4s %-10s %-7s %9s %9s | %9s %9s %9s\n' $L $rt $place \
        "$ov" "$ba" "$re" "$wo" "$fj"
      echo "$ov $ba" > "$R/one-$L-$rt-$place"
    done
  done
done

note "nested, outer x inner threads, us per outer region (median of 5 runs of 50000)"
echo "Each outer member forks an inner team that runs one loop and its barrier."
echo "overhead = region - work; work = one inner thread's share of the loop."
echo "libomp runs twice: by default, which creates every inner team again, and"
echo "with hot inner teams, which it keeps as pomp does. Hot teams get full inner"
echo "teams only without OMP_THREAD_LIMIT (else libomp gives them 1 thread), so"
echo "libomp runs without it. Pinned, every inner thread is on its own CPU, the"
echo "same one for both runtimes. Any short team is flagged: that timing is"
echo "invalid."
printf '%-6s %-12s %-7s %9s | %9s %9s\n' shape runtime place overhead region work
short=0
for s in $NESTS; do
  n1=${s%x*}
  n2=${s#*x}
  for rt in $RUNTIMES $([ $HAVE_LIBOMP = 1 ] && echo libomp-hot); do
    case $rt in
    pomp*) RE="OMP_THREAD_LIMIT=$((n1 * n2))" ;;
    libomp) RE= ;;
    libomp-hot) RE="KMP_HOT_TEAMS_MAX_LEVEL=2 KMP_HOT_TEAMS_MODE=1" ;;
    esac
    X=$(exe ${rt%-hot})
    for place in free pinned; do
      P=$([ $place = pinned ] && echo "POMP_TEST_CPUS=$CPUS")
      E="env $LIBENV OMP_MAX_ACTIVE_LEVELS=2 $RE $P"
      v=$(runs5 $E "$X" nestbench 50000 $n1 $n2)
      re=$(echo "$v" | med_of region)
      wo=$(echo "$v" | med_of work)
      ov=$(echo "$v" | med_of overhead)
      flag=
      if echo "$v" | grep -qE 'short teams ([1-9][0-9]* outer|[0-9]+ outer [1-9])'; then
        flag="  SHORT TEAMS: invalid"
        short=$((short + 1))
        ov=-
      fi
      printf '%-6s %-12s %-7s %9s | %9s %9s%s\n' $s $rt $place "$ov" "$re" "$wo" "$flag"
      echo "$ov" > "$R/nest-$s-$rt-$place"
    done
  done
done
[ $short = 0 ] || echo "$short nested timings had short teams and are invalid"

if [ $HAVE_LIBOMP = 1 ]; then
  note "summary: the two metrics, pomp against libomp (us; ratio = libomp/pomp)"
  echo "overhead: of a parallel region with one static loop; barrier: one barrier."
  for metric in overhead barrier; do
    for place in free pinned; do
      echo
      echo "$metric, $place"
      printf '%-6s %8s' T libomp
      for p in $POMPS; do printf ' | %10s %6s' $p ratio; done
      echo
      for L in $LIMITS; do
        read lo lb < "$R/one-$L-libomp-$place"
        l=$([ $metric = overhead ] && echo $lo || echo $lb)
        printf '%-6s %8s' $L "$l"
        for p in $POMPS; do
          read po pb < "$R/one-$L-$p-$place"
          v=$([ $metric = overhead ] && echo $po || echo $pb)
          printf ' | %10s %6s' "$v" "$(ratio "$v" "$l")"
        done
        echo
      done
      [ $metric = overhead ] || continue
      for s in $NESTS; do
        l=$(cat "$R/nest-$s-libomp-hot-$place")
        printf '%-6s %8s' $s "$l"
        for p in $POMPS; do
          v=$(cat "$R/nest-$s-$p-$place")
          printf ' | %10s %6s' "$v" "$(ratio "$v" "$l")"
        done
        echo
      done
    done
  done
  echo
  echo "Nested rows compare with libomp-hot (libomp keeping its inner teams);"
  echo "plain libomp, which builds them again each region, is in the table above."
fi

note "pomp_zbench, sections 4 and 6 (fork/join designs; F is pomp today)"
ZB_TS=$(echo $LIMITS | tr ' ' ',')
POMP_ZB_SECTIONS=4,6 "$B/pomp_zbench" $ZB_TS 2>&1 | sed -n '/^4\./,/^done/p' | sed '$d'
note "pomp_zbench, sections 4 and 6, body = 2000 iterations"
POMP_ZB_SECTIONS=4,6 POMP_ZB_WORK=2000 "$B/pomp_zbench" $ZB_TS 2>&1 |
  sed -n '/^4\./,/^done/p' | sed '$d'

note "summary: $fail failures"
exit $((fail != 0))
