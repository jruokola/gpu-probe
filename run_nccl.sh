#!/usr/bin/env bash
set -euo pipefail

# ── 1. Where the archive lives ───────────────────────────────────────────────
# Build it once on a machine that has git:
#   git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
#   cd nccl-tests
#   git archive --format=tar.gz --prefix=nccl-tests/ HEAD  \
#               -o /opt/nccl-tests/nccl-tests-2.20.tar.gz
# Then bake /opt/nccl-tests/nccl-tests-2.20.tar.gz into the container image
# or copy it to a shared path accessible by all Slurm nodes.
SRC_TARBALL="/app/nccl-tests-2.20.tar.gz"

# ── 2. Scratch build dir ─────────────────────────────────────────────────────
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "Extracting NCCL-tests -> $BUILD_DIR"
tar -xzf "$SRC_TARBALL" -C "$BUILD_DIR"
cd "$BUILD_DIR"/nccl-tests*

# ── 3. Build (single-GPU MPI-less binary) ────────────────────────────────────
make MPI=0

# ── 4. Run all_reduce_perf on ONE GPU for 30 s (8 MiB payload) ───────────────
./build/all_reduce_perf -b 8M -e 8M -f 2 -g 1 | tee /nccl.txt

echo -e "\n--- NCCL Test Output (/nccl.txt) ---"
cat /tmp/nccl.txt
echo "----------------------------------------"

bw=$(grep "8.0M" /nccl.txt | awk '{print $(NF-1)}')
if [[ -z "$bw" ]]; then
    echo "ERROR: failed to parse bandwidth"
    exit 1
fi
echo "Measured NCCL bandwidth: ${bw} GB/s"
# Uncomment to enforce a pass/fail threshold
# (( $(echo "$bw < 90" | bc -l) )) && { echo "NCCL bandwidth too low"; exit 1; }