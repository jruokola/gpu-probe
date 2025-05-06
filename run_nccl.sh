#!/usr/bin/env bash
set -e
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
make MPI=0
./build/all_reduce_perf -b 8M -e 8M -f 2 -g 2 | tee /tmp/nccl.txt
bw=$(grep "8.0M" /tmp/nccl.txt | awk '{print $(NF-1)}')
echo "NCCL bw ${bw} GB/s"
(( $(echo "$bw < 90" | bc -l) )) && { echo "NCCL bandwidth too low"; exit 1; }