#!/bin/bash
export PYXIS_VERBOSE=3
#SBATCH --job-name=granite-dist-ft
#SBATCH --nodes=2                   # Utilize 2 H100 nodes
#SBATCH --ntasks-per-node=1         # 1 task (PyTorch process) per node, each managing one H100
#SBATCH --gres=gpu:1                # Each task gets 1 H100 GPU
#SBATCH --cpus-per-task=16          # Allocate more CPUs, e.g., 16 per H100 for data loading, etc.
#SBATCH --mem=180G                  # Memory per node (adjusted from 200G to fit node's ~187GiB allocatable)
#SBATCH --time=04:00:00             # Example: 4 hours wall time for a distributed job
#SBATCH --output=/root/gpu-probe_%j.log # Log file in /root on login node (accessible on shared storage)
#SBATCH --export=ALL # Export environment variables to srun tasks

# --- Soperator Specifics - THESE ARE CRITICAL --- 
# You MUST find out from Soperator/Nebius documentation how to specify:
# 1. The container image to use (you'll need to build one for gpu-probe)
# 2. How shared filesystems (if any are strictly needed by probe scripts beyond /tmp) are mounted.
# 3. The working directory if not /app or / (where scripts are).

# Example placeholder for Soperator (syntax is LIKELY DIFFERENT):
#SBATCH --container-image=cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/gpu-probe:latest 
#SBATCH --container-workdir=/app # Assuming /app is WORKDIR in Dockerfile for gpu-probe
#SBATCH --container-mounts=/:/mnt/shared_fs_in_container

echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID (gpu-probe-multi) on node: $SLURMD_NODENAME"
echo "Requesting $SLURM_GPUS_ON_NODE GPUs (as per sbatch --gres)."

# Set environment variables that runner.py might need
# These should ideally be set by the user submitting the job or via a wrapper script if they vary.
export MLFLOW_TRACKING_URI="public-tracking.mlflow-e00spsxj9fckt2k6pz.backbone-e00qwnewt2vzn1dh4s.msp.eu-north1.nebius.cloud"
export MLFLOW_TRACKING_SERVER_CERT_PATH="/etc/mlflow/certs/ca.pem"
export MLFLOW_EXPERIMENT_NAME="GPU_Probes_MultiGPU" # Or your desired experiment name
export MLFLOW_TRACKING_USERNAME="jruokola" # If using authenticated MLflow
export MLFLOW_TRACKING_PASSWORD="K3m1k44l1!!666"

# CUDA_VISIBLE_DEVICES should be automatically set by Slurm based on --gres
# The sub-scripts (gpu-burn, nccl) should pick this up.

echo "Running src/gpu_probe/runner.py --test ..."

# Ensure PYTHONPATH includes the location of the gpu_probe package if necessary
# If WORKDIR is /app and your package is in /app/src/gpu_probe, then:
# export PYTHONPATH=/app/src:$PYTHONPATH 

# srun will execute this command on the allocated node, inside the container environment
srun python -m src.gpu_probe.runner --test

JOB_EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID (gpu-probe-multi) finished with exit code $JOB_EXIT_CODE."
exit $JOB_EXIT_CODE 