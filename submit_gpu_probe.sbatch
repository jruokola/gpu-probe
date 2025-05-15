#!/bin/bash
#SBATCH --job-name=gpu-probe-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 1 container/GPU per node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/root/gpu_probe_%j.log
#SBATCH --export=ALL                 # let Pyxis propagate rank vars

# ─── Cluster / Pyxis tuning ────────────────────────────────────────────────
export PYXIS_VERBOSE=3
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# give rendez-vous more time on a fresh multi-node launch
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_TIMEOUT=900        # seconds

# ─── Image + shared mount point ------------------------------------------------
IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/gpu-probe-4:latest"
HOST_MNT_BASE="/data/cifar10_gpu_probe"     # *must exist on every node*
CNTR_MNT_BASE="/app/dataset"                # path *inside* the container

# make sure the host directory exists (safe even if it already does)
srun --ntasks=$SLURM_NNODES --nodes=$SLURM_NNODES \
     bash -c "mkdir -p $HOST_MNT_BASE && chmod 777 $HOST_MNT_BASE"

# ─── Master address & port -----------------------------------------------------
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_IP=$(getent hosts $MASTER_NODE | awk '{print $1}')
MASTER_PORT=$((49152 + RANDOM % 16384))

echo "MASTER: $MASTER_NODE  ($MASTER_IP:$MASTER_PORT)"
echo "IMAGE : $IMAGE"
echo "MOUNT : $HOST_MNT_BASE  ->  $CNTR_MNT_BASE"

# ─── 1. Node-local GPU sanity checks (only on master) -------------------------
srun --nodes=1 --ntasks=1 --exact --nodelist=$MASTER_NODE \
     --container-image=$IMAGE \
     --container-workdir=/app \
     --container-mounts="$HOST_MNT_BASE:$CNTR_MNT_BASE:rw" \
     python -m gpu_probe.runner --test
PROBE_RC=$?

# ─── 2. Multi-node CIFAR-10 micro-benchmark -----------------------------------
TRAIN_ARGS="--epochs 1 --batches_per_epoch 20 \
            --lr 0.01 --data_path $CNTR_MNT_BASE"

TORCHRUN_CMD="torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
  src/gpu_probe/train.py $TRAIN_ARGS"

srun --container-image=$IMAGE \
     --container-workdir=/app \
     --container-env=MASTER_ADDR=$MASTER_IP,MASTER_PORT=$MASTER_PORT \
     --container-mounts="$HOST_MNT_BASE:$CNTR_MNT_BASE:rw" \
     bash -c "$TORCHRUN_CMD"
TRAIN_RC=$?

# ─── 3. Final status -----------------------------------------------------------
if [[ $PROBE_RC -eq 0 && $TRAIN_RC -eq 0 ]]; then
    echo "✅  Job finished successfully."
    exit 0
else
    echo "❌  Job failed (probe=$PROBE_RC , train=$TRAIN_RC)."
    exit 1
fi