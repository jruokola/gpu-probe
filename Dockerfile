# ── 1. CUDA + PyTorch base image ────────────────────────────────
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git, make, gcc needed by run_*.sh scripts and potentially some pip packages
RUN apt-get update && apt-get install -y git make gcc --no-install-recommends && rm -rf /var/lib/apt/lists/*

# ── 3. Copy project & resolve deps with pip ─────────────────────
WORKDIR /app

# ----------  Python deps via pip  ----------
COPY requirements.txt .
# COPY pyproject.toml . # Keep if other tools use it, or remove if only for uv

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project context
# This includes src/, run_*.sh, entrypoint.sh, etc.
COPY . .

# Add /app/src to PYTHONPATH so `python -m gpu_probe.X` works, assuming 'gpu_probe' is a package inside 'src'.
# If 'gpu_probe' is directly in /app (i.e. /app/gpu_probe/runner.py), then PYTHONPATH="/app:${PYTHONPATH}" was correct.
# Based on your file structure attachment (gpu-probe/src/gpu_probe), /app/src is correct.
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Ensure the scripts are executable
RUN chmod +x entrypoint.sh run_nccl.sh run_gpu_burn.sh src/gpu_probe/runner.py src/gpu_probe/train.py

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--test"]
