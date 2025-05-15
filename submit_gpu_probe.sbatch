#!/bin/bash
#SBATCH --job-name=gpu-probe-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one container / GPU per node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/root/gpu_probe_%j.log
#SBATCH --export=ALL               # let Pyxis export RANK/LOCAL_RANK/…

export PYXIS_VERBOSE=3

# ─── constants ───────────────────────────────────────────────────────────────
DOCKER_IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/gpu-probe-4:latest"
SHARED_DATA="/cifar10_gpu_probe_data"              # one dir visible to all nodes

FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
FIRST_IP=$(getent hosts "$FIRST_NODE" | awk '{print $1}')

# ─── 1. node-local GPU sanity checks on the first node ───────────────────────
srun --nodes=1 --ntasks=1 --exact --nodelist="$FIRST_NODE" \
     --container-image="$DOCKER_IMAGE" \
     --container-workdir=/app \
     --container-mounts="${SHARED_DATA}:${SHARED_DATA}:rw" \
     python -m gpu_probe.runner --test
PROBE_RC=$?

# ─── 2. multi-node ResNet-50 / CIFAR-10 micro-run ────────────────────────────
export MASTER_ADDR="$FIRST_IP"
export MASTER_PORT=$((49152 + RANDOM % 16384))     # random high port

TRAIN_ARGS="--epochs 1 --batches_per_epoch 20 --lr 0.01 --data_path ${SHARED_DATA}"

SRUN_CMD="
torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  src/gpu_probe/train.py ${TRAIN_ARGS}
"

srun --container-image="$DOCKER_IMAGE" \
     --container-workdir=/app \
     --container-env=MASTER_ADDR=${MASTER_ADDR} \
     --container-env=MASTER_PORT=${MASTER_PORT} \
     --container-mounts="${SHARED_DATA}:${SHARED_DATA}:rw" \
     bash -c "$SRUN_CMD"
TRAIN_RC=$?

# ─── 3. final status ─────────────────────────────────────────────────────────
if [[ $PROBE_RC -eq 0 && $TRAIN_RC -eq 0 ]]; then
    echo "✅  All GPU-probe steps completed successfully."
    exit 0
else
    echo "❌  One or more steps failed (runner=$PROBE_RC, train=$TRAIN_RC)."
    exit 1
fi