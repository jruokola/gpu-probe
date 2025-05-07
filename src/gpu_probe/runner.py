#!/usr/bin/env python
import os
import subprocess
import mlflow
import logging
import re
import sys

# Define package-relative paths if needed, but absolute paths work in container
# Assuming WORKDIR in Dockerfile is /app and gpu_probe package is in /app/src/gpu_probe
# Or if WORKDIR is /app/src/gpu_probe, then paths might be relative or simpler.
# For scripts at root of image:
NCCL_SCRIPT = "/app/run_nccl.sh"
GPU_BURN_SCRIPT = "/app/run_gpu_burn.sh"
# Command to run the training module - now launched by torchrun
# TRAIN_COMMAND = "python -m gpu_probe.train"
NCCL_OUTPUT_FILE = "/tmp/nccl.txt"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_command(command, script_name, is_train_test=False):
    """Runs a shell command and logs its output and exit code."""
    logging.info(f"Running {script_name}...")
    logging.info(f"Executing command: {command}")
    try:
        # For torchrun, we need to ensure it can find other ranks.
        # The necessary env vars (MASTER_ADDR, etc.) should be set by Slurm and passed by srun --export=ALL
        # If sub-launching torchrun from a python script run by srun, it inherits these.
        process = subprocess.run(
            command,
            shell=True,  # shell=True needed for complex commands like torchrun with args
            check=False,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )
        output = f"""{process.stdout}\n{process.stderr}"""
        exit_code = process.returncode
        logging.info(f"{script_name} finished with exit code: {exit_code}")
        if mlflow.active_run():
            mlflow.log_text(output, f"{script_name}_output.txt")
            mlflow.log_metric(f"{script_name}_exit_code", exit_code)
        return exit_code, output
    except Exception as e:
        logging.error(f"Failed to run {script_name}: {e}")
        if mlflow.active_run():
            mlflow.log_metric(f"{script_name}_exit_code", -1)
            mlflow.log_text(str(e), f"{script_name}_error.txt")
        return -1, str(e)


def parse_nccl_bandwidth(output_file):
    """Parses the NCCL bandwidth from the output file."""
    try:
        with open(output_file, "r") as f:
            content = f.read()
        match = re.search(
            r"8\.0M\s+.*?(\d+\.?\d*)\s+(\d+\.?\d*)$", content, re.MULTILINE
        )
        if match:
            bandwidth = float(match.group(1))
            logging.info(f"Parsed NCCL bandwidth: {bandwidth} GB/s")
            return bandwidth
        else:
            logging.warning(
                f"Could not parse bandwidth using regex from {output_file}. Content:\n{content}"
            )
            match_echo = re.search(r"NCCL bw (\d+\.?\d*) GB/s", content)
            if match_echo:
                bandwidth = float(match_echo.group(1))
                logging.info(f"Parsed NCCL bandwidth from echo: {bandwidth} GB/s")
                return bandwidth
            else:
                logging.warning("Could not parse bandwidth from echo line either.")
                return None
    except FileNotFoundError:
        logging.error(f"NCCL output file not found: {output_file}")
        return None
    except Exception as e:
        logging.error(f"Error parsing NCCL output file {output_file}: {e}")
        return None


def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    if not test_mode:
        logging.warning("Runner called without --test argument. Exiting.")
        sys.exit(0)

    # Determine number of GPUs available from Slurm environment (if set by sbatch --gres)
    # PyTorch DDP typically relies on LOCAL_RANK, RANK, WORLD_SIZE set by launcher.
    # torchrun will handle setting these for the train.py processes.
    # We need to tell torchrun how many processes per node (gpus per node).
    num_gpus_per_node = os.getenv(
        "SLURM_GPUS_ON_NODE", os.getenv("SLURM_JOB_GPUS", "1")
    ).split("(")[0]
    try:
        num_gpus_per_node = int(num_gpus_per_node)
    except ValueError:
        gpu_match = re.search(r"gpu:(\d+)", num_gpus_per_node)  # e.g. from gpu:tesla:2
        if gpu_match:
            num_gpus_per_node = int(gpu_match.group(1))
        else:
            num_gpus_per_node = 1  # Default to 1 if parsing fails
    logging.info(
        f"Detected/Defaulting to {num_gpus_per_node} GPUs per node for torchrun."
    )

    # Training command construction for distributed training using torchrun
    # It assumes train.py is written to use torch.distributed and env:// init method.
    # The main runner.py (this script) is launched once by srun.
    # This single runner.py process then launches the distributed train.py using torchrun.
    train_script_path = "-m gpu_probe.train"  # Using module invocation
    train_command = (
        f"torchrun "
        f"--nproc_per_node={num_gpus_per_node} "
        f"--nnodes=1 "  # Since runner.py is on one node, train.py runs on this node's GPUs
        f"--rdzv_id=$SLURM_JOB_ID "  # Use Slurm job ID for rendezvous
        f"--rdzv_backend=c10d "
        f"--rdzv_endpoint=$SLURMD_NODENAME:29500 "  # Master addr is current node, choose a port
        f"{train_script_path} "
        f"--epochs 1 --batches_per_epoch 10 --no_mlflow"  # Quick test for train component
    )

    try:
        with mlflow.start_run() as run:
            logging.info(f"Started MLflow run: {run.info.run_id}")
            node_name = os.getenv(
                "SLURMD_NODENAME", os.getenv("NODE_NAME", "unknown_node")
            )
            slurm_job_id = os.getenv("SLURM_JOB_ID", "N/A")
            mlflow.set_tag("node_name", node_name)
            mlflow.set_tag("slurm_job_id", slurm_job_id)
            mlflow.log_param("test_type", "gpu_probe_slurm_runner")

            nccl_exit_code, _ = run_command(NCCL_SCRIPT, "nccl_test")
            if nccl_exit_code == 0:
                bandwidth = parse_nccl_bandwidth(NCCL_OUTPUT_FILE)
                if bandwidth is not None:
                    mlflow.log_metric("nccl_bandwidth_gbs", bandwidth)

            gpu_burn_exit_code, _ = run_command(GPU_BURN_SCRIPT, "gpu_burn_test")
            mlflow.log_metric("gpu_burn_passed", 1 if gpu_burn_exit_code == 0 else 0)

            # Run the training test (which is now a distributed DDP job on the node's GPUs)
            train_exit_code, train_output = run_command(
                train_command, "train_test_distributed", is_train_test=True
            )
            mlflow.log_metric(
                "train_test_dist_passed", 1 if train_exit_code == 0 else 0
            )
            if train_exit_code == 0:
                match_throughput = re.search(
                    r"imgs/s: (\d+)", train_output
                )  # Assuming train.py prints this
                if match_throughput:
                    mlflow.log_metric(
                        "train_throughput_imgs_s", int(match_throughput.group(1))
                    )

            logging.info("MLflow run completed for gpu_probe.")
    except Exception as e:
        logging.error(f"MLflow tracking or script execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
