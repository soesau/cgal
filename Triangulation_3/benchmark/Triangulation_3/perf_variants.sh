#!/usr/bin/env bash
# Phase 1 performance variants of Delaunay_triangulation_3 insertion.
#
# For handle and index storage and each macro set below, this script
#   1. builds DT3_benchmark and the correctness check (variants/),
#   2. checks the triangulation (is_valid + fingerprint, must match "base"),
#   3. measures with perf stat, pinned to one P-core,
# and prints a summary table relative to "base".
#
# Usage: bash perf_variants.sh [points_file] [iterations] [repetitions]
# Results: perf_results/variants/

cd "$(dirname "$0")" || exit 1

DATA="${1:-Nefertiti.xyz}"
ITER="${2:-10}"
REPS="${3:-3}"
CPU=0
CGAL_ROOT=/home/oesi/Code/t3i
OUT=perf_results/variants
VAR=variants
mkdir -p "$OUT" "$VAR"

declare -A CFG=(
  [base]=""
  [index]="-DCGAL_TDS3_BRANCHLESS_INDEX"
  [hole]="-DCGAL_TDS3_SMALL_HOLE_DIRECTED_EDGES"
  [unroll]="-DCGAL_T3_FIND_CONFLICTS_UNROLL"
  [all]="-DCGAL_TDS3_BRANCHLESS_INDEX -DCGAL_TDS3_SMALL_HOLE_DIRECTED_EDGES -DCGAL_T3_FIND_CONFLICTS_UNROLL"
)
ORDER=(base index hole unroll all)
declare -A STORE=([h]="-DHANDLE_STORAGE" [i]="-DINDEX_STORAGE")
STORES=(h i)

build() { # <build dir> <source dir> <target> <cxx flags>
  cmake -S "$2" -B "$1" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCGAL_DIR="$CGAL_ROOT" -DCMAKE_CXX_FLAGS="$4" > "$1.cmake.log" 2>&1 \
  && ninja -C "$1" "$3" > "$1.build.log" 2>&1
}

: > "$OUT/check.txt"
declare -A BUILT

# ---------------------------------------------------------------- build + check
for s in "${STORES[@]}"; do
  for c in "${ORDER[@]}"; do
    name="${s}_${c}"
    flags="${STORE[$s]} ${CFG[$c]}"
    echo "=== build $name  ($flags)"
    if ! build "$VAR/build_$name" . DT3_benchmark "$flags"; then
      echo "!! build of DT3_benchmark failed for $name, see $VAR/build_$name.build.log"
      continue
    fi
    cp "$VAR/build_$name/DT3_benchmark" "$VAR/DT3_$name"
    if ! build "$VAR/check_$name" check DT3_check "$flags"; then
      echo "!! build of DT3_check failed for $name, see $VAR/check_$name.build.log"
      continue
    fi
    echo "=== check $name"
    res=$("$VAR/check_$name/DT3_check" "$DATA")
    echo "$name $res" | tee -a "$OUT/check.txt"
    BUILT[$name]=1
  done
done

# Compare fingerprints with the base variant of the same storage
echo
ok=1
for s in "${STORES[@]}"; do
  ref=$(grep "^${s}_base " "$OUT/check.txt" | sed 's/^[^ ]* //')
  for c in "${ORDER[@]}"; do
    line=$(grep "^${s}_${c} " "$OUT/check.txt" | sed 's/^[^ ]* //')
    [[ -z "$line" ]] && continue
    if [[ "$line" != "$ref" || "$line" != valid=yes* ]]; then
      echo "!! ${s}_${c} differs from ${s}_base or is invalid:"
      echo "     base: $ref"
      echo "     this: $line"
      ok=0
    fi
  done
done
[[ $ok == 1 ]] && echo "All variants valid and identical to their base triangulation."

# ---------------------------------------------------------------------- perf
echo
for s in "${STORES[@]}"; do
  for c in "${ORDER[@]}"; do
    name="${s}_${c}"
    [[ -z "${BUILT[$name]}" ]] && continue
    echo "=== perf $name"
    taskset -c "$CPU" perf stat -r "$REPS" -x, \
        -e duration_time,cycles,instructions,branch-misses \
        -o "$OUT/$name.csv" "$VAR/DT3_$name" "$DATA" "$ITER" > /dev/null \
      || echo "!! perf failed for $name"
  done
done

# ------------------------------------------------------------------- summary
python3 - "$OUT" "${STORES[@]}" -- "${ORDER[@]}" <<'EOF' | tee "$OUT/summary.txt"
import sys, os
out = sys.argv[1]
sep = sys.argv.index('--')
stores, order = sys.argv[2:sep], sys.argv[sep+1:]

def read(name):
    p = os.path.join(out, name + '.csv')
    if not os.path.exists(p):
        return None
    vals = {}
    for line in open(p):
        f = line.strip().split(',')
        if len(f) < 3 or line.startswith('#'):
            continue
        ev = f[2]
        if ev.startswith('cpu_atom'):
            continue
        ev = ev.replace('cpu_core/', '').strip('/')
        try:
            vals[ev] = float(f[0])
        except ValueError:
            pass
    return vals

cols = [('duration_time', 'time [s]', 1e-9), ('cycles', 'cycles [G]', 1e-9),
        ('instructions', 'instr [G]', 1e-9), ('branch-misses', 'br-miss [M]', 1e-6)]
for s in stores:
    base = read(f'{s}_base')
    print(f"\n{'handle' if s == 'h' else 'index'} storage")
    print(f"{'variant':10s}" + ''.join(f'{t:>22s}' for _, t, _ in cols))
    for c in order:
        v = read(f'{s}_{c}')
        if v is None:
            continue
        row = f'{c:10s}'
        for ev, _, scale in cols:
            x = v.get(ev)
            if x is None:
                row += f"{'n/a':>22s}"; continue
            rel = ''
            if base and base.get(ev) and c != 'base':
                rel = f' ({100*(x/base[ev]-1):+.1f}%)'
            row += f'{x*scale:>13.3f}{rel:>9s}'
        print(row)
EOF
