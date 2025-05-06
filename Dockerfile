# ── 1. CUDA + PyTorch base image ────────────────────────────────
FROM nvcr.io/nvidia/pytorch:24.07-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# ── 2. Install uv binary (12 KiB) ───────────────────────────────
RUN curl -Ls https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

# Install git, make, gcc needed by run_*.sh scripts
RUN apt-get update && apt-get install -y git make gcc --no-install-recommends && rm -rf /var/lib/apt/lists/*

# ── 3. Copy project & resolve deps with uv ─────────────────────
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

# Copy certificate into image
# Adjust path relative to the Docker build context (gpu-probe/)
COPY /mlflow-cert/ca.pem /etc/mlflow/certs/ca.pem

# Set environment variable for MLflow client to find the cert
ENV MLFLOW_TRACKING_SERVER_CERT_PATH=/etc/mlflow/certs/ca.pem

# ----------  Python deps  ----------
# Copy pyproject.toml to the current directory
COPY pyproject.toml .
# Sync dependencies using uv based on pyproject.toml
# This installs dependencies listed under [project.dependencies]
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Copy the entire project context
# This includes src/, run_*.sh, entrypoint.sh, etc.
COPY . .

# Ensure the scripts are executable (uv sync might not preserve permissions)
# Use relative paths within WORKDIR (/app)
RUN chmod +x entrypoint.sh run_nccl.sh run_gpu_burn.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--test"]