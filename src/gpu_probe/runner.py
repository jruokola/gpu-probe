#!/usr/bin/env python
import logging
import os
import re
import subprocess
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
        # Log output to console or a local file if needed, MLflow is removed
        logging.debug(f"{script_name} output:\n{output}")
        return exit_code, output
    except Exception as e:
        logging.error(f"Failed to run {script_name}: {e}")
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
        node_name = os.getenv("SLURMD_NODENAME", os.getenv("NODE_NAME", "unknown_node"))
        slurm_job_id = os.getenv("SLURM_JOB_ID", "N/A")
        slurm_step_id = os.getenv("SLURM_STEP_ID", "N/A")

        logging.info(f"Starting node-local GPU probe on node: {node_name}")
        logging.info(f"SLURM Job ID: {slurm_job_id}, Step ID: {slurm_step_id}")
        logging.info("Test type: gpu_probe_node_local_tests")

        # --- Run NCCL Test (node-local) ---
        # Assumes run_nccl.sh uses GPUs available on this node.
        nccl_exit_code, _ = run_command(NCCL_SCRIPT, "nccl_test_local")
        if nccl_exit_code == 0:
            bandwidth = parse_nccl_bandwidth(NCCL_OUTPUT_FILE)
            if bandwidth is not None:
                logging.info(f"NCCL Bandwidth (local): {bandwidth} GB/s")
            else:
                logging.warning("Failed to parse NCCL bandwidth (local).")
        else:
            logging.error("NCCL test (local) failed.")

        # --- Run GPU Burn Test (node-local) ---
        # Assumes run_gpu_burn.sh uses GPUs available on this node.
        gpu_burn_exit_code, _ = run_command(GPU_BURN_SCRIPT, "gpu_burn_test_local")
        if gpu_burn_exit_code == 0:
            logging.info("GPU Burn test (local) passed.")
        else:
            logging.error("GPU Burn test (local) failed.")

        # --- Training test is now a separate Slurm job step for multi-node ---
        logging.info("Node-local tests (NCCL, GPU Burn) completed.")
        logging.info(
            "Multi-node distributed training test for train.py should be a separate srun command in the sbatch script."
        )

        # Determine overall success for exit code
        if nccl_exit_code == 0 and gpu_burn_exit_code == 0:
            logging.info("All node-local GPU probe tests passed.")
            sys.exit(0)
        else:
            logging.error("One or more node-local GPU probe tests failed.")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Runner script execution failed: {e}")
        sys.exit(1)  # Exit with error if runner itself fails


if __name__ == "__main__":
    main()
