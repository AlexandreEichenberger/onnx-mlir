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
#   ZB_MAP=1                            # also run pomp_zbench sections 1, 2
#   PLACE_TRACE=1                       # also trace where threads run
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
# pomp with sleep and wake-up counters, for diagnosis only (not timed).
mkdir -p "$B/stats"
$CXX $CXXOPT -DPOMP_STATS -Wall -c pomp.cpp -o "$B/stats/pomp.o" &&
  $CC "$B/pomp_test.o" "$B/stats/pomp.o" -pthread -o "$B/stats/pomp_test" ||
  { echo "build of pomp-stats failed"; exit 1; }

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
  for S in 0 5 200000 auto; do
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
  run "$T/pomp_host_test" options $L
  run env OMP_NUM_THREADS=3 POMP_SPIN=7 "$T/pomp_host_test" options $L
  run env OMP_NUM_THREADS=3 POMP_SPIN=7 POMP_IGNORE_ENV=1 "$T/pomp_host_test" options $L
done
echo "$runs runs, $fail failures"

# Timings: median of 5 runs. libomp spins 200 ms before sleeping; pomp
# adapts its spin per thread (POMP_SPIN=auto, 2-100 us). Every timing starts after 1000 untimed regions
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
rm -rf "$R"; mkdir -p "$R"  # stale rows would enter the best-of

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

# The spin time of pomp's waits against the cost of regions where nothing
# computes while threads wait: back-to-back regions, nested regions.
note "pomp spin (POMP_SPIN: auto, or us), us per region (median of 5 runs of 20000)"
echo "One level: overhead and fork/join, at T=8 and 16. Nested: overhead, 4x4 and 8x8."
printf '%-8s %9s %9s %9s %9s %9s %9s\n' POMP_SPIN "ov T=8" "fj T=8" "ov T=16" "fj T=16" \
  "nest 4x4" "nest 8x8"
for S in auto 0 5 20 50 100 200000; do
  row=
  for L in 8 16; do
    E="env OMP_THREAD_LIMIT=$L OMP_NUM_THREADS=$L POMP_SPIN=$S"
    row="$row $(runs5 $E "$(exe pomp)" loop 20000 | med_of overhead)"
    row="$row $(runs5 $E "$(exe pomp)" bench 20000 | med_of join)"
  done
  for sh in 4x4 8x8; do
    n1=${sh%x*} n2=${sh#*x}
    row="$row $(runs5 env POMP_SPIN=$S OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=$((n1 * n2)) \
      "$(exe pomp)" nestbench 20000 $n1 $n2 | med_of overhead)"
  done
  printf '%-8s %9s %9s %9s %9s %9s %9s\n' $S $row
done

note "pomp sleeps and wake-ups per region (pomp built with counters; one run each)"
echo "idle: a worker slept waiting for the next region; team: a join or barrier"
echo "wait slept; posts: wake-ups sent (at forks: by the leader's fork)."
for S in auto 5 20 100 200000; do
  for w in "bench 8" "bench 16" "nest 8x8"; do
    set -- $w
    if [ $1 = nest ]; then
      n1=${2%x*} n2=${2#*x}
      out=$(env POMP_SPIN=$S OMP_MAX_ACTIVE_LEVELS=2 OMP_THREAD_LIMIT=$((n1 * n2)) \
        "$B/stats/pomp_test" nestbench 20000 $n1 $n2 2>&1)
    else
      out=$(env POMP_SPIN=$S OMP_THREAD_LIMIT=$2 OMP_NUM_THREADS=$2 \
        "$B/stats/pomp_test" bench 20000 2>&1)
    fi
    echo "POMP_SPIN=$S $1 $2: $(echo "$out" | sed -n 's/^stats: //p')"
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

# One row of the one-level table, saved for the best-of summary. ONE_PRE, if
# set, is a command to run the test under (taskset).
ONE_PRE=
one_row() { # T runtime place env-assignments...
  local L=$1 rt=$2 place=$3 X b lo
  shift 3
  X=$(exe $rt)
  b=$(runs5 $ONE_PRE env $LIBENV OMP_THREAD_LIMIT=$L OMP_NUM_THREADS=$L "$@" "$X" bench 100000)
  lo=$(runs5 $ONE_PRE env $LIBENV OMP_THREAD_LIMIT=$L OMP_NUM_THREADS=$L "$@" "$X" loop 100000)
  local fj ba re wo ov
  fj=$(echo "$b" | med_of join)
  ba=$(echo "$b" | med_of barrier)
  re=$(echo "$lo" | med_of region)
  wo=$(echo "$lo" | med_of work)
  ov=$(echo "$lo" | med_of overhead)
  printf '%-4s %-10s %-13s %9s %9s | %9s %9s %9s\n' $L $rt $place \
    "$ov" "$ba" "$re" "$wo" "$fj"
  echo "$ov $ba" > "$R/one-$L-$rt-$place"
}

# Placement, from the Linux topology: CPUs as a list; a CPU's core and chip.
expand() { echo "$1" | tr ',' '\n' | while IFS=- read a b; do seq $a ${b:-$a}; done; }
topo() { cat /sys/devices/system/cpu/cpu$1/topology/$2 2>/dev/null; }
chip_of() { echo "$(topo $1 physical_package_id).$(topo $1 book_id).$(topo $1 drawer_id)"; }
if [ -r /sys/devices/system/cpu/cpu0/topology/core_id ]; then
  ALL=$(expand "$CPUS")
  C0=$(echo "$ALL" | head -1)
  # One CPU per core, in order: the first SMT thread of each core.
  PERCORE=$(for c in $ALL; do
    f=$(topo $c thread_siblings_list | sed 's/[-,].*//')
    [ "$f" = "$c" ] && echo $c
  done | paste -sd, -)
  # The CPUs of C0's chip, and one per core of them.
  CHIP=$(for c in $ALL; do [ "$(chip_of $c)" = "$(chip_of $C0)" ] && echo $c; done |
    paste -sd, -)
  CHIP1=$(echo "$PERCORE" | tr ',' '\n' | while read c; do
    [ "$(chip_of $c)" = "$(chip_of $C0)" ] && echo $c; done | paste -sd, -)
  NCHIP=$(echo "$CHIP" | tr ',' '\n' | wc -l)
  NCHIP1=$(echo "$CHIP1" | tr ',' '\n' | wc -l)
  # A partner for CPU C0: its SMT sibling, a CPU of another core of its
  # chip, and a CPU of another chip.
  SIB= NEAR= FAR=
  for c in $ALL; do
    [ $c = $C0 ] && continue
    if [ "$(chip_of $c)" != "$(chip_of $C0)" ]; then
      FAR=${FAR:-$c}
    elif [ "$(topo $c core_id)" = "$(topo $C0 core_id)" ]; then
      SIB=${SIB:-$c}
    else
      NEAR=${NEAR:-$c}
    fi
  done

  note "placement variants, one level, us per region (median of 5 runs of 100000)"
  echo "pinned1: thread t on the t-th of one CPU per core, $(echo $PERCORE | cut -c1-40)..."
  echo "chip-free: not pinned, but the process limited (taskset) to the leader's"
  echo "  chip, CPUs $CHIP; only for T <= $NCHIP."
  echo "chip1-free: the same, one CPU per core of the chip, $CHIP1; T <= $NCHIP1."
  echo "cores-close, cores-spread: libomp's own binding, OMP_PLACES=cores with"
  echo "OMP_PROC_BIND=close or spread (pomp has no such binding)."
  echo "The free and pinned (two threads per core) rows above count in the best-of below."
  printf '%-4s %-10s %-13s %9s %9s | %9s %9s %9s\n' T runtime place \
    overhead barrier region work fork/join
  for L in $LIMITS; do
    for rt in $RUNTIMES; do
      one_row $L $rt pinned1 POMP_TEST_CPUS=$PERCORE
      if command -v taskset >/dev/null; then
        if [ $L -le $NCHIP ]; then
          ONE_PRE="taskset -c $CHIP" one_row $L $rt chip-free
        fi
        if [ $L -le $NCHIP1 ]; then
          ONE_PRE="taskset -c $CHIP1" one_row $L $rt chip1-free
        fi
      fi
      if [ $rt = libomp ]; then
        one_row $L $rt cores-close OMP_PLACES=cores OMP_PROC_BIND=close
        one_row $L $rt cores-spread OMP_PLACES=cores OMP_PROC_BIND=spread
      fi
    done
  done
  if [ $HAVE_LIBOMP = 1 ]; then
    echo
    echo "libomp's binding at T=4 (OMP_DISPLAY_AFFINITY), close then spread:"
    for b in close spread; do
      env $LIBENV OMP_THREAD_LIMIT=4 OMP_NUM_THREADS=4 OMP_PLACES=cores \
        OMP_PROC_BIND=$b OMP_DISPLAY_AFFINITY=true \
        OMP_AFFINITY_FORMAT="  $b: thread %n on CPUs %A" \
        "$(exe libomp)" bench 10 2>&1 | grep "^  $b:" | sort -u
    done

    note "best placement of each runtime (us; ratio = libomp/pomp)"
    echo "Each runtime at its own best place, among those above."
    for metric in overhead barrier; do
      echo
      echo "$metric"
      printf '%-4s %10s %-13s | %10s %-13s %6s\n' T libomp place pomp place ratio
      for L in $LIMITS; do
        set -- $(for rt in libomp pomp; do
          for f in "$R"/one-$L-$rt-*; do
            read o b < "$f"
            v=$([ $metric = overhead ] && echo $o || echo $b)
            echo "$v ${f##*/one-$L-$rt-}"
          done | sort -n | head -1
        done)
        printf '%-4s %10s %-13s | %10s %-13s %6s\n' $L $1 $2 $3 $4 "$(ratio $3 $1)"
      done
    done
  fi

  note "T=2, worker placed by distance from the leader (CPU $C0), us (median of 5)"
  echo "Both threads pinned. sibling: the leader's SMT sibling; chip: another core"
  echo "of its chip; other: another chip."
  printf '%-10s %-8s %5s %9s %9s %9s %9s\n' runtime worker cpu overhead fork/join barrier work
  for rt in $RUNTIMES; do
    for w in sibling:$SIB chip:$NEAR other:$FAR; do
      c=${w#*:}
      [ -n "$c" ] || continue
      E="env $LIBENV OMP_THREAD_LIMIT=2 OMP_NUM_THREADS=2 POMP_TEST_CPUS=$C0,$c"
      b=$(runs5 $E "$(exe $rt)" bench 100000)
      lo=$(runs5 $E "$(exe $rt)" loop 100000)
      printf '%-10s %-8s %5s %9s %9s %9s %9s\n' $rt ${w%%:*} $c \
        "$(echo "$lo" | med_of overhead)" "$(echo "$b" | med_of join)" \
        "$(echo "$b" | med_of barrier)" "$(echo "$lo" | med_of work)"
    done
  done

  # SMT interference at a join or a barrier, with uneven work: does a thread
  # that waits slow the work of its SMT sibling more than spinning saves?
  note "uneven work: a waiting thread next to a computing one, us per phase (median of 5)"
  echo "Even tids run LONG passes of work, odd tids SHORT (one pass: 1024 floats,"
  echo "about 0.08 us), then all meet; join,ldr-short swaps them, so the leader,"
  echo "short, waits at the join next to a long worker."
  echo "  join: one region per phase. barrier: 10 barriers per region."
  echo "  sibling: tids 2k, 2k+1 on the two SMT threads of one core (CPUs ${C0},${SIB}...);"
  echo "  own core: every thread on a core of its own, same chip."
  echo "  alone: the LONG work with one thread. excess = phase - alone: the runtime's"
  echo "  overhead plus the slowdown of the long work by its waiting sibling."
  echo "Wait policies: pomp POMP_SPIN = auto (default: adapts per thread, 2-100 us),"
  echo "or a fixed number of us spinning before sleeping (0 sleeps at once);"
  echo "libomp KMP_BLOCKTIME = time spinning before sleeping."
  # CPUs for T threads: pairs of SMT siblings, or one per core, on C0's chip.
  PAIRS=$(for c in $(echo "$CHIP1" | tr ',' ' '); do
    echo "$c,$(expand "$(topo $c thread_siblings_list)" | grep -vx $c | head -1)"
  done | paste -sd, -)
  printf '%-2s %-8s %-14s %-9s %-6s %-20s %9s %9s\n' T place sync long/shrt runtime \
    policy phase excess
  for sz in 250/25/1000 2500/250/100; do
    lg=$(echo $sz | cut -d/ -f1) sh=$(echo $sz | cut -d/ -f2) it=$(echo $sz | cut -d/ -f3)
    for rt in $RUNTIMES; do
      eval "alone_$rt=\$(runs5 env $LIBENV OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1 \
        POMP_TEST_CPUS=$C0 \"\$(exe $rt)\" imbal $it $lg $sh | med_of phase)"
    done
    for T in 2 4; do
      for place in sibling own-core; do
        if [ $place = sibling ]; then
          P=$(echo "$PAIRS" | cut -d, -f1-$T)
        else
          P=$(echo "$CHIP1" | cut -d, -f1-$T)
        fi
        [ $(echo "$P" | tr ',' '\n' | grep -c .) = $T ] || continue
        for sync in "join even" "join odd" "barrier even"; do
          case $sync in
          "join even") sname=join ;;
          "join odd") sname=join,ldr-short ;;
          *) sname=barrier ;;
          esac
          for rt in $RUNTIMES; do
            if [ $rt = pomp ]; then
              POLS="POMP_SPIN=auto POMP_SPIN=200000 POMP_SPIN=20 POMP_SPIN=5 POMP_SPIN=0"
            else
              POLS="KMP_BLOCKTIME=200 KMP_BLOCKTIME=20us KMP_BLOCKTIME=0"
            fi
            eval "al=\$alone_$rt"
            for pol in $POLS; do
              ph=$(runs5 env $LIBENV OMP_THREAD_LIMIT=$T OMP_NUM_THREADS=$T \
                POMP_TEST_CPUS=$P $pol "$(exe $rt)" imbal $it $lg $sh $sync |
                med_of phase)
              ex=$(awk -v a="$ph" -v b="$al" 'BEGIN { printf "%.2f", a - b }')
              printf '%-2s %-8s %-14s %-9s %-6s %-20s %9s %9s\n' $T $place \
                $sname $lg/$sh $rt $pol "$ph" "$ex"
            done
          done
        done
      done
    done
    for rt in $RUNTIMES; do
      eval "echo \"alone, $lg passes, $rt: \$alone_$rt us\""
    done
  done
fi

# Where free and pinned threads run (section 6 of POMP_Z_Tuning.md), settled:
# PLACE_TRACE=1 runs it again.
if [ "${PLACE_TRACE:-0}" = 1 ]; then
  note "placement: where threads run, empty fork/join (one run of 100000 per row)"
  echo "timed: each region timed alone. traced: each thread records its CPU in"
  echo "each region (slower), and regions are grouped by the farthest worker from"
  echo "the leader. Context switches are over the timed pass, all threads."
  for L in $LIMITS; do
    for rt in $RUNTIMES; do
      for place in free pinned; do
        P=$([ $place = pinned ] && echo "POMP_TEST_CPUS=$CPUS")
        echo "-- T=$L $rt $place"
        env $LIBENV OMP_THREAD_LIMIT=$L OMP_NUM_THREADS=$L $P "$(exe $rt)" place 100000 2>&1
      done
    done
  done
fi

ZB_TS=$(echo $LIMITS | tr ' ' ',')
# Sections 1 and 2 (atomics, line transfer by topology) are settled: see
# POMP_Z_Tuning.md. ZB_MAP=1 runs them again.
if [ "${ZB_MAP:-0}" = 1 ]; then
  note "pomp_zbench, sections 1 and 2 (atomics; line transfer by topology)"
  POMP_ZB_SECTIONS=1,2 "$B/pomp_zbench" $ZB_TS 2>&1 | sed -n '/^1\./,/^done/p' | sed '$d'
fi
note "pomp_zbench, sections 4 and 6 (fork/join designs; F is pomp today)"
POMP_ZB_SECTIONS=4,6 "$B/pomp_zbench" $ZB_TS 2>&1 | sed -n '/^4\./,/^done/p' | sed '$d'
note "pomp_zbench, section 8 (waking sleeping workers)"
POMP_ZB_SECTIONS=8 "$B/pomp_zbench" $ZB_TS 2>&1 | sed -n '/^8\./,/^done/p' | sed '$d'
note "pomp_zbench, section 7 (a waiting thread next to a computing one)"
POMP_ZB_SECTIONS=7 "$B/pomp_zbench" 2>&1 | sed -n '/^7\./,/^done/p' | sed '$d'
note "pomp_zbench, section 4, body = 2000 iterations"
POMP_ZB_SECTIONS=4 POMP_ZB_WORK=2000 "$B/pomp_zbench" $ZB_TS 2>&1 |
  sed -n '/^4\./,/^done/p' | sed '$d'

note "summary: $fail failures"
exit $((fail != 0))
