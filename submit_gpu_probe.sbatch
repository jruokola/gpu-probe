#!/bin/bash
export PYXIS_VERBOSE=3 # Enable Pyxis verbose logging
#SBATCH --job-name=gpu-probe-multi-node
#SBATCH --nodes=2                   # Request 2 nodes for multi-node training test
#SBATCH --ntasks-per-node=1         # 1 task per node for the distributed training
#SBATCH --gres=gpu:1                # 1 GPU per task/node
#SBATCH --cpus-per-task=16          # CPUs for each task (train.py process)
#SBATCH --mem=64G                   # Memory per node (train.py uses CIFAR10, moderate memory)
#SBATCH --time=01:00:00             # Time limit for all tests on one node
#SBATCH --output=/root/gpu_probe_multinode_%j.log # Log file
#SBATCH --export=ALL                # Export environment variables to srun tasks

# --- Host Path for Shared Storage (where logs might go if /root is part of it) ---
HOST_SHARED_FS_ROOT_PATH="/" # Assuming filestore is at / on host
HOST_JOB_OUTPUTS_BASE_PATH="${HOST_SHARED_FS_ROOT_PATH}/root/slurm_jobs_output/gpu_probe_multinode_${SLURM_JOB_ID}"

echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID (gpu-probe-multi-node) on node: $SLURMD_NODENAME"
echo "SLURM_PROCID=$SLURM_PROCID, SLURM_NPROCS=$SLURM_NPROCS, SLURM_JOB_GPUS=$SLURM_JOB_GPUS, SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

# MLflow Configuration (exported to srun via --export=ALL)
export MLFLOW_TRACKING_URI="public-tracking.mlflow-e00spsxj9fckt2k6pz.backbone-e00qwnewt2vzn1dh4s.msp.eu-north1.nebius.cloud"
export MLFLOW_TRACKING_SERVER_CERT_PATH="/etc/mlflow/certs/ca.pem" # Path inside container
export MLFLOW_EXPERIMENT_NAME="GPU_Probe_MultiNode_Tests"
export MLFLOW_TRACKING_USERNAME="jruokola"
export MLFLOW_TRACKING_PASSWORD="K3m1k44l1!!666"

# The runner.py script uses /tmp for nccl.txt, which is ephemeral to the container.
# No explicit data/checkpoint mounts needed for gpu-probe itself, unless its sub-scripts require them.

# The main command to be executed inside the container by srun
# runner.py will internally use torchrun for the train.py part to utilize the allocated GPUs.
# PYTHONPATH is set in the Dockerfile to /app/src, WORKDIR is /app.
RUNNER_COMMAND="python -m gpu_probe.runner --test"

echo "------------------------------------------------------------------------------"
echo "Step 1: Running Node-Local Probes (NCCL, GPU Burn) on FIRST allocated node: $SLURMD_NODENAME"
echo "(runner.py will log its own MLflow run for this step)"
# Use srun to restrict runner.py to only the first node of the allocation for these local tests
# It will use the 1 GPU allocated to task 0 on that node.
SRUN_RUNNER_CMD="python -m gpu_probe.runner --test"
srun --nodes=1 --ntasks=1 --exact --nodelist=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
    --container-image="docker://<YOUR_REGISTRY>/gpu-probe:latest" \
    --container-workdir="/app" \
    $SRUN_RUNNER_CMD
RUNNER_EXIT_CODE=$?
echo "Node-local probes (runner.py) on first node finished with exit code $RUNNER_EXIT_CODE."
echo "------------------------------------------------------------------------------"

echo "------------------------------------------------------------------------------"
echo "Step 2: Launching Multi-Node Distributed Training Test (gpu_probe.train.py)"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29501 # Use a different port just in case
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT (for torch.distributed.run)"

# Arguments for train.py for this multi-node test
# --no_mlflow because runner.py already started an MLflow run for the overall job (or train.py can log to a new nested run)
# Let's have train.py log to a new run for clarity of the distributed step
TRAIN_ARGS="--epochs 1 --batches_per_epoch 20 --lr 0.01 --data_path /tmp/cifar10_dist_probe"
TRAIN_ARGS+=" --output_dir /tmp/train_probe_output_${SLURM_JOB_ID}" # Ephemeral output inside container

echo "Train arguments: ${TRAIN_ARGS}"

# Use srun to launch python -m torch.distributed.run across ALL allocated nodes ($SLURM_NNODES)
# Each srun task will be one process in the distributed training job.
# The --container-image, workdir, mounts are inherited by srun if not overridden, but explicit is safer.

SRUN_TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=1 --nnodes=${SLURM_NNODES} --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} gpu_probe.train ${TRAIN_ARGS}"
echo "Executing on all ${SLURM_NNODES} nodes: ${SRUN_TRAIN_CMD}"

srun \
    --container-image="docker://<YOUR_REGISTRY>/gpu-probe:latest" \
    --container-workdir="/app" \
    bash -c "${SRUN_TRAIN_CMD}" # Wrap in bash -c to handle complex command with args

TRAIN_EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
echo "Multi-node distributed training test finished with exit code $TRAIN_EXIT_CODE."

if [ "$RUNNER_EXIT_CODE" -ne 0 ] || [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo "One or more GPU probe steps failed."
    exit 1
else
    echo "All GPU probe steps completed successfully."
    exit 0
fi 