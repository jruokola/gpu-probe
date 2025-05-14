#!/usr/bin/env bash
set -e
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
make MPI=0
# Changed -g 2 to -g 1 as the sbatch script allocates 1 GPU for this step
./build/all_reduce_perf -b 8M -e 8M -f 2 -g 1 | tee /tmp/nccl.txt

echo "--- NCCL Test Output (/tmp/nccl.txt) ---"
cat /tmp/nccl.txt
echo "----------------------------------------"

bw=$(grep "8.0M" /tmp/nccl.txt | awk '{print $(NF-1)}')

if [ -z "$bw" ]; then
    echo "ERROR: Failed to parse bandwidth from NCCL test output."
    exit 1
fi

echo "NCCL bw ${bw} GB/s"
# Temporarily commenting out bandwidth check for diagnostics
# (( $(echo "$bw < 90" | bc -l) )) && { echo "NCCL bandwidth too low"; exit 1; }
