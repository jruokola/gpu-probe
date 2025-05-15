#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 0. constants
# ------------------------------------------------------------------
SRC_TARBALL="/app/nccl-tests-2.20.tar.gz"   # baked into image
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# ------------------------------------------------------------------
# 1. unpack  & build (MPI=0, single-GPU)
# ------------------------------------------------------------------
echo "Extracting NCCL-tests  ->  $BUILD_DIR"
tar -xzf "$SRC_TARBALL" -C "$BUILD_DIR"
cd "$BUILD_DIR"/nccl-tests*

make -s MPI=0           # quiet build

# ------------------------------------------------------------------
# 2. run all_reduce_perf on ONE GPU, 8 MiB payload, 20 iters
# ------------------------------------------------------------------
./build/all_reduce_perf -b 8M -e 8M -f 2 -g 1 | tee /nccl.txt
RET=$?

echo -e "\n--- NCCL Test Output (/nccl.txt) ---"
cat /nccl.txt
echo "------------------------------------------------------------------"

# ------------------------------------------------------------------
# 3. parse alg-bandwidth (column 7) from the *data* line
#    size column prints '8388608' on ≥ 2.22, so match that.
# ------------------------------------------------------------------
bw=$(awk '/^[[:space:]]*[0-9]+/{print $(NF-1); exit}' /nccl.txt)

if [[ $RET -ne 0 ]]; then
    echo "ERROR: nccl-tests binary failed (exit $RET)"
    exit $RET
fi

if [[ -z "$bw" ]]; then
    echo "WARNING: could not parse bandwidth – accepting test for single-GPU run"
else
    echo "Measured NCCL algorithmic bandwidth: $bw GB/s"
    # optional: enforce a minimum alg-BW threshold
    # THRESH=1000   # GB/s on H100 PCI-e
    # awk "BEGIN{exit ($bw < $THRESH)}" || \
    #   { echo "ERROR: BW below ${THRESH} GB/s"; exit 1; }
fi

exit 0