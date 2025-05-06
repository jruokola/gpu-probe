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
NCCL_SCRIPT = "/run_nccl.sh"
GPU_BURN_SCRIPT = "/run_gpu_burn.sh"
# Command to run the training module from /app (if src is where gpu_probe package is)
# If WORKDIR is /app/src/gpu_probe and train.py is in the same dir, it could be "python train.py"
TRAIN_COMMAND = "python -m gpu_probe.train"  # Assumes gpu_probe is in PYTHONPATH
NCCL_OUTPUT_FILE = "/tmp/nccl.txt"  # As defined in run_nccl.sh

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_command(command, script_name):
    """Runs a shell command and logs its output and exit code."""
    logging.info(f"Running {script_name}...")
    logging.info(f"Executing command: {command}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )
        output = f"""{process.stdout}\n{process.stderr}"""
        exit_code = process.returncode
        logging.info(f"{script_name} finished with exit code: {exit_code}")
        if mlflow.active_run():  # Check if MLflow run is active before logging
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

    try:
        with mlflow.start_run() as run:  # Ensure run context is managed
            logging.info(f"Started MLflow run: {run.info.run_id}")

            # Use SLURMD_NODENAME if available (from Slurm), otherwise NODE_NAME (from K8s downward API), then fallback
            node_name = os.getenv(
                "SLURMD_NODENAME", os.getenv("NODE_NAME", "unknown_node")
            )
            slurm_job_id = os.getenv("SLURM_JOB_ID", "N/A")

            username = os.getenv("MLFLOW_TRACKING_USERNAME", "unknown_user")
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Default_GPU_Probe")

            mlflow.set_tag("node_name", node_name)
            mlflow.set_tag("slurm_job_id", slurm_job_id)
            mlflow.log_param("tracking_username", username)
            mlflow.log_param("experiment_name_env", experiment_name)
            mlflow.log_param("test_type", "gpu_probe_via_slurm")

            # --- Run NCCL Test ---
            nccl_exit_code, _ = run_command(NCCL_SCRIPT, "nccl_test")
            if nccl_exit_code == 0:
                bandwidth = parse_nccl_bandwidth(NCCL_OUTPUT_FILE)
                if bandwidth is not None:
                    mlflow.log_metric("nccl_bandwidth_gbs", bandwidth)
                else:
                    logging.warning("Could not log NCCL bandwidth metric.")
            else:
                logging.error(f"NCCL test failed with exit code {nccl_exit_code}")

            # --- Run GPU Burn Test ---
            gpu_burn_exit_code, _ = run_command(GPU_BURN_SCRIPT, "gpu_burn_test")
            mlflow.log_metric("gpu_burn_passed", 1 if gpu_burn_exit_code == 0 else 0)
            if gpu_burn_exit_code != 0:
                logging.error(
                    f"GPU Burn test failed with exit code {gpu_burn_exit_code}"
                )

            # --- Run Train Test ---
            train_exit_code, train_output = run_command(TRAIN_COMMAND, "train_test")
            mlflow.log_metric("train_test_passed", 1 if train_exit_code == 0 else 0)
            if train_exit_code == 0:
                match_throughput = re.search(r"imgs/s: (\d+)", train_output)
                if match_throughput:
                    throughput = int(match_throughput.group(1))
                    mlflow.log_metric("train_throughput_imgs_s", throughput)
                    logging.info(f"Logged training throughput: {throughput} imgs/s")
                else:
                    logging.warning("Could not parse training throughput from output.")
            else:
                logging.error(f"Train test failed with exit code {train_exit_code}")

            logging.info("MLflow run completed for gpu_probe.")

    except Exception as e:
        logging.error(f"MLflow tracking or script execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
