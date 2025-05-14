#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Settings – change only these two paths
# ---------------------------------------------------------------------------
SRC_TARBALL="/app/gpu-burn-1.0.tar.gz"   # pre-downloaded archive
BUILD_DIR="$(mktemp -d)"                          # scratch build dir

# ---------------------------------------------------------------------------
# 1. unpack source
# ---------------------------------------------------------------------------
if [[ ! -f "$SRC_TARBALL" ]]; then
  echo "ERROR: gpu-burn tarball not found at $SRC_TARBALL" >&2
  exit 1
fi
echo "Extracting gpu-burn from $SRC_TARBALL → $BUILD_DIR"
tar -xzf "$SRC_TARBALL" -C "$BUILD_DIR"

cd "$BUILD_DIR"/gpu-burn*              # handles versioned dir names

# ---------------------------------------------------------------------------
# 2. build
# ---------------------------------------------------------------------------
echo "Building gpu-burn..."
make  # assumes CUDA toolkit & build-essentials already present

# ---------------------------------------------------------------------------
# 3. run burn test (30 s default or use $1)
# ---------------------------------------------------------------------------
SECONDS=${1:-30}
echo "Running gpu-burn for $SECONDS seconds..."
./gpu_burn "$SECONDS"

# ---------------------------------------------------------------------------
# 4. cleanup
# ---------------------------------------------------------------------------
echo "gpu-burn completed; cleaning up."
rm -rf "$BUILD_DIR"
exit 0