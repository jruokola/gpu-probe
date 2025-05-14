# GPU Probe Suite

This suite of scripts is designed to perform stress testing and functional checks on GPUs within a containerized environment, typically managed by Slurm. It includes node-local hardware tests (GPU burn, NCCL bandwidth) and a multi-node distributed PyTorch training test.

## Components

- **`Dockerfile`**: Defines the container image with necessary dependencies including PyTorch, CUDA, NCCL, and tools like `git`, `make`, `gcc` required by the test scripts.
- **`requirements.txt`**: Lists Python dependencies (e.g., `torch`, `torchvision`).
- **`src/gpu_probe/runner.py`**: Python script that orchestrates node-local tests. It executes `run_nccl.sh` and `run_gpu_burn.sh`.
- **`src/gpu_probe/train.py`**: A PyTorch script that performs a simple distributed training routine (CIFAR10 dataset with ResNet50 model) using `DistributedDataParallel (DDP)` to verify multi-GPU/multi-node training functionality.
- **`run_nccl.sh`**: A shell script that clones the NVIDIA `nccl-tests` repository, builds them, and runs `all_reduce_perf` to measure inter-GPU communication bandwidth on a node.
- **`run_gpu_burn.sh`**: A shell script that clones the `gpu-burn` utility, builds it, and runs it to stress-test GPUs for stability and thermal performance. The default duration is 30 seconds.
- **`entrypoint.sh`**: The main entrypoint for the Docker container, which executes `src/gpu_probe/runner.py`.
- **`submit_gpu_probe.sbatch`**: An example Slurm batch script that demonstrates how to run the GPU probe suite. It includes steps for:
    1. Running node-local probes (`runner.py`) on the first allocated node.
    2. Running the distributed training test (`train.py`) across all allocated nodes and GPUs using `torchrun`.

## Prerequisites

- **Docker**: For building the container image.
- **Slurm Workload Manager**: For submitting and managing the job on a cluster.
- **Pyxis/Enroot (or similar Slurm container integration)**: For running Docker images with Slurm (`srun --container-image=...`).
- **NVIDIA GPUs**: Accessible to the Slurm cluster nodes.
- **Container Registry**: (Optional, if not using local image files like `.sqsh`) To host the built Docker image.

## Build Instructions

1. Navigate to the `gpu-probe` directory.
2. Build the Docker image:

    ```bash
    docker build -t your-registry/your-repo/gpu-probe:latest -f Dockerfile .
    ```

    Replace `your-registry/your-repo/gpu-probe:latest` with your desired image name and tag.

## Running the Probe Suite

1. **Push Image (if using a registry):**
    If you're using a central container registry, push the built image:

    ```bash
    docker push your-registry/your-repo/gpu-probe:latest
    ```

2. **Update sbatch Script:**
    Edit `submit_gpu_probe.sbatch`:
    - Set the `DOCKER_IMAGE` variable to the correct path of your image in the registry (e.g., `DOCKER_IMAGE=your-registry/your-repo/gpu-probe:latest`).
    - If you are using a locally converted Enroot image (`.sqsh` file), update `DOCKER_IMAGE` to the absolute path of the `.sqsh` file on the cluster nodes (e.g., `DOCKER_IMAGE=/path/to/shared/gpu-probe.sqsh`).
    - Adjust Slurm parameters (`--nodes`, `--gres=gpu:X`, `--mem`, `--time`, `--output`) as needed for your cluster and testing requirements.
    - Modify `TRAIN_ARGS` if you need different training parameters or a different shared `--data_path` for the CIFAR10 dataset. Ensure the chosen `--data_path` is on a shared filesystem accessible by all nodes with the same path.

3. **Submit the Slurm Job:**

    ```bash
    sbatch submit_gpu_probe.sbatch
    ```

## Workflow Overview

The `submit_gpu_probe.sbatch` script orchestrates the following:

1. **Node-Local Probe (on first node):**
    - `srun` launches the container on the first allocated node.
    - The container's `entrypoint.sh` executes `python -m gpu_probe.runner --test`.
    - `runner.py` then executes:
        - `run_nccl.sh`: Clones, builds, and runs `nccl-tests` (`all_reduce_perf` on 1 GPU by default). Output and parsed bandwidth are logged.
        - `run_gpu_burn.sh`: Clones, builds, and runs `gpu-burn`.
    - The exit code of `runner.py` (`PROBE_RC`) indicates success (0) or failure (1) of these local tests.

2. **Multi-Node Distributed Training:**
    - `srun` launches `torchrun` across all allocated nodes and GPUs.
    - `torchrun` executes `src/gpu_probe/train.py` for a DDP training session.
    - The script downloads CIFAR10 to the specified `--data_path` (master rank downloads, others wait) and trains a ResNet50 model for a few epochs/batches.
    - The exit code of this step (`TRAIN_RC`) indicates success or failure.

3. **Final Result:**
    - The sbatch script checks `PROBE_RC` and `TRAIN_RC` to determine overall success or failure of the probe.

## Output and Logging

- **Slurm Output**: The main log file is specified by `--output` in `submit_gpu_probe.sbatch` (default: `/root/gpu_probe_%j.log`). This captures stdout/stderr from the sbatch script itself and the `srun` commands.
- **Script Logs**: `runner.py` and `train.py` use Python's `logging` module, which prints to stdout/stderr and will be captured in the Slurm output file.
- **NCCL Test Output**: `run_nccl.sh` saves the raw output of `all_reduce_perf` to `/tmp/nccl.txt` *inside the container* and also prints this content to stdout.

## Customization

- **GPU Burn Duration**: Modify the `gpu_burn 30` command in `run_gpu_burn.sh` (30 is seconds).
- **NCCL Test Parameters**: Adjust the `all_reduce_perf` arguments in `run_nccl.sh` (e.g., `-b`, `-e`, `-g`). The current bandwidth check in `run_nccl.sh` is commented out for diagnostic purposes but can be re-enabled and its threshold adjusted.
- **Training Parameters**: Modify `TRAIN_ARGS` in `submit_gpu_probe.sbatch` to change epochs, batch count, learning rate, or data path for `train.py`.
- **Container Base Image**: The `Dockerfile` uses `nvcr.io/nvidia/pytorch:24.07-py3`. This can be changed if a different PyTorch/CUDA version is needed.
- **Python Dependencies**: Add to `requirements.txt` and rebuild the Docker image.
