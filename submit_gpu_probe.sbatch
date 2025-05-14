#!/bin/bash
#SBATCH --job-name=gpu-probe-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/root/gpu_probe_%j.log
#SBATCH --export=ALL

export PYXIS_VERBOSE=3

# -------------------------------------------------------------------------
# 1. user-level configuration
# -------------------------------------------------------------------------
DOCKER_IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/gpu-probe:latest"
SHARED_DATA="/cifar10_gpu_probe_data"                 # one directory for all nodes

# -------------------------------------------------------------------------
# 2. node-local probe on the first node only
# -------------------------------------------------------------------------
FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --nodes=1 --ntasks=1 --exact --nodelist="$FIRST_NODE" \
     --container-image="$DOCKER_IMAGE" \
     --container-workdir=/app \
     --container-mounts="$SHARED_DATA:$SHARED_DATA:rw" \
     python -m gpu_probe.runner --test
PROBE_RC=$?

# -------------------------------------------------------------------------
# 3. multi-node distributed ResNet-50 / CIFAR-10 training
# -------------------------------------------------------------------------
export MASTER_ADDR="$FIRST_NODE"
export MASTER_PORT=29501

GPUS_PER_NODE=$SLURM_GPUS_ON_NODE        # =1 in this batch
WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))

TRAIN_ARGS="--epochs 1 --batches_per_epoch 20 --lr 0.01 --data_path $SHARED_DATA"

SRUN_CMD="torchrun \
          --nnodes=$SLURM_NNODES \
          --nproc_per_node=$GPUS_PER_NODE \
          --rdzv_backend=c10d \
          --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
          --node_rank=\${SLURM_NODEID} \
          src/gpu_probe/train.py ${TRAIN_ARGS}"

srun --container-image="$DOCKER_IMAGE" \
     --container-workdir=/app \
     --container-env=MASTER_ADDR,MASTER_PORT \
     --container-mounts="$SHARED_DATA:$SHARED_DATA:rw" \
     bash -c "$SRUN_CMD"
TRAIN_RC=$?

# -------------------------------------------------------------------------
# 4. final result
# -------------------------------------------------------------------------
if [[ $PROBE_RC -eq 0 && $TRAIN_RC -eq 0 ]]; then
    echo "✅  All GPU-probe steps completed successfully."
    exit 0
else
    echo "❌  One or more steps failed (runner=$PROBE_RC, train=$TRAIN_RC)."
    exit 1
fi