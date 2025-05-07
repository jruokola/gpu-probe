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
        logging.warning(
            "Runner called without --test argument. This script is a simple orchestrator. Exiting."
        )
        sys.exit(0)

    # This runner will now only perform node-local tests (NCCL, GPU Burn).
    # Distributed training test for train.py will be a separate srun step in the sbatch script.

    try:
        with mlflow.start_run() as run:  # MLflow run for this node's local probe tests
            logging.info(
                f"Started MLflow run for node-local GPU probe: {run.info.run_id}"
            )
            node_name = os.getenv(
                "SLURMD_NODENAME", os.getenv("NODE_NAME", "unknown_node")
            )
            slurm_job_id = os.getenv(
                "SLURM_JOB_ID", "N/A"
            )  # This will be the main job ID
            slurm_step_id = os.getenv(
                "SLURM_STEP_ID", "N/A"
            )  # If runner.py is launched as a step

            mlflow.set_tag("node_name", node_name)
            mlflow.set_tag("slurm_job_id", slurm_job_id)
            mlflow.set_tag("slurm_step_id", slurm_step_id)
            mlflow.log_param("test_type", "gpu_probe_node_local_tests")

            # --- Run NCCL Test (node-local) ---
            # Assumes run_nccl.sh uses GPUs available on this node.
            nccl_exit_code, _ = run_command(NCCL_SCRIPT, "nccl_test_local")
            if nccl_exit_code == 0:
                bandwidth = parse_nccl_bandwidth(NCCL_OUTPUT_FILE)
                if bandwidth is not None:
                    mlflow.log_metric("nccl_bandwidth_gbs_local", bandwidth)

            # --- Run GPU Burn Test (node-local) ---
            # Assumes run_gpu_burn.sh uses GPUs available on this node.
            gpu_burn_exit_code, _ = run_command(GPU_BURN_SCRIPT, "gpu_burn_test_local")
            mlflow.log_metric(
                "gpu_burn_passed_local", 1 if gpu_burn_exit_code == 0 else 0
            )

            # --- Training test is now a separate Slurm job step for multi-node ---
            logging.info("Node-local tests (NCCL, GPU Burn) completed.")
            logging.info(
                "Multi-node distributed training test for train.py should be a separate srun command in the sbatch script."
            )

            logging.info("MLflow run for node-local GPU probe completed.")

    except Exception as e:
        logging.error(f"MLflow tracking or local script execution failed: {e}")
        # Decide on exit code strategy here. If this runner is just one part of a larger sbatch script,
        # its exit might not terminate the whole sbatch job unless sbatch is configured to do so.
        sys.exit(1)  # Exit with error if runner itself fails


if __name__ == "__main__":
    main()
