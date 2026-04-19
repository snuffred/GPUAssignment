#!/bin/bash
#
# Auto-test all .in files under test_files/.
# For each test: runs flood_seq and flood_cuda on a DAS-5 compute node,
# checks correctness with the provided Python script, and prints speedup.
#
# Usage:
#   ./run_all_tests.sh                 # test all files in test_files/
#   ./run_all_tests.sh debug small_mountains    # test only the named files
#
# Requirements: prun available, flood_seq and flood_cuda built (run `make all`).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_DIR="test_files"
OUT_DIR="${OUT_DIR:-./test_outputs}"
SUMMARY_FILE="$OUT_DIR/summary.txt"
mkdir -p "$OUT_DIR"

PRUN_OPTS=(-t 15:00 -np 1 -native '-C gpunode')

if [[ ! -x ./flood_seq || ! -x ./flood_cuda ]]; then
    echo "Binaries missing. Running 'make all'..."
    make all >/dev/null || { echo "Build failed."; exit 1; }
fi

# Mirror all stdout to the summary file (per-test outputs are written directly to $OUT_DIR).
exec > >(tee "$SUMMARY_FILE")

# Collect test names
if [[ $# -gt 0 ]]; then
    tests=("$@")
else
    tests=()
    for f in "$TEST_DIR"/*.in; do
        tests+=("$(basename "$f" .in)")
    done
fi

# Extract one field from the "Result:" line (1-indexed across the 7 fields)
get_field() {
    grep "^Result:" "$1" | sed 's/^Result: *//' | awk -F',' -v i="$2" '{gsub(/^ +| +$/,"",$i); print $i}'
}
get_time() {
    grep "^Time:" "$1" | awk '{print $2}'
}

printf "\n%-22s | %10s | %10s | %8s | %s\n" "Test" "Seq (s)" "CUDA (s)" "Speedup" "Correctness"
printf "%s\n" "---------------------------------------------------------------------------"

total_seq=0
total_cuda=0
pass=0
fail=0

for name in "${tests[@]}"; do
    infile="$TEST_DIR/$name.in"
    if [[ ! -f "$infile" ]]; then
        printf "%-22s | %s\n" "$name" "input file not found: $infile"
        continue
    fi

    args=$(cat "$infile")
    seq_out="$OUT_DIR/${name}.seq.out"
    cuda_out="$OUT_DIR/${name}.cuda.out"

    prun "${PRUN_OPTS[@]}" ./flood_seq  $args > "$seq_out"  2>/dev/null
    prun "${PRUN_OPTS[@]}" ./flood_cuda $args > "$cuda_out" 2>/dev/null

    t_seq=$(get_time "$seq_out")
    t_cuda=$(get_time "$cuda_out")

    if [[ -z "$t_seq" || -z "$t_cuda" ]]; then
        printf "%-22s | %s\n" "$name" "FAILED to run (see $seq_out / $cuda_out)"
        fail=$((fail + 1))
        continue
    fi

    speedup=$(awk -v a="$t_seq" -v b="$t_cuda" 'BEGIN{ if (b>0) printf "%.2fx", a/b; else print "inf" }')

    check=$(python3 "$TEST_DIR/check_correctness.py" "$seq_out" "$cuda_out")
    if [[ "$check" == *"matches"* ]]; then
        status="PASS"
        pass=$((pass + 1))
    else
        status="FAIL"
        fail=$((fail + 1))
    fi

    printf "%-22s | %10s | %10s | %8s | %s\n" "$name" "$t_seq" "$t_cuda" "$speedup" "$status"

    total_seq=$(awk -v a="$total_seq" -v b="$t_seq"  'BEGIN{print a+b}')
    total_cuda=$(awk -v a="$total_cuda" -v b="$t_cuda" 'BEGIN{print a+b}')
done

printf "%s\n" "---------------------------------------------------------------------------"
if awk "BEGIN{exit !($total_cuda > 0)}"; then
    overall=$(awk -v a="$total_seq" -v b="$total_cuda" 'BEGIN{printf "%.2fx", a/b}')
else
    overall="n/a"
fi
printf "%-22s | %10.4f | %10.4f | %8s | %d pass, %d fail\n" \
       "TOTAL" "$total_seq" "$total_cuda" "$overall" "$pass" "$fail"

printf "\nOutputs saved to: %s\n" "$OUT_DIR"

# Exit non-zero if any test failed, so CI / callers can detect it.
[[ $fail -eq 0 ]]
