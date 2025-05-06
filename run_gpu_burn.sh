#!/usr/bin/env bash
set -e

# Clone, build and run gpu-burn directly in the current environment
# Assumes git, make, gcc are installed
echo "Cloning gpu-burn..."
git clone https://github.com/wilicc/gpu-burn || { echo 'Failed to clone gpu-burn'; exit 1; }

cd gpu-burn || { echo 'Failed cd to gpu-burn'; exit 1; }

echo "Building gpu-burn..."
make || { echo 'Failed to build gpu-burn'; exit 1; }

echo "Running gpu-burn..."
./gpu_burn 30 || { echo 'gpu-burn command failed'; exit 1; }

echo "gpu-burn completed."
# Cleanup (optional)
cd ..
rm -rf gpu-burn

exit 0