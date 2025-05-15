#!/bin/bash
#SBATCH --job-name=gpu-probe-multinode
#SBATCH --nodes=2                  # 2× H100 nodes
#SBATCH --ntasks-per-node=1        # 1 container / GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/root/gpu_probe_%j.log
#SBATCH --export=ALL               # expose RANK, LOCAL_RANK, …

###################### cluster / NCCL tweaks ######################
export PYXIS_VERBOSE=3
export TORCH_DISTRIBUTED_TIMEOUT=900
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker

###################### image + shared mount #######################
IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/gpu-probe-9:latest"
HOST_CACHE="/data/cifar10_gpu_probe"      # exists on *all* hosts
CNTR_CACHE="/app/dataset"                 # path *inside* container

###################### prepare host cache dir #####################
srun --ntasks=$SLURM_NNODES bash -c "mkdir -p $HOST_CACHE && chmod 777 $HOST_CACHE"

###################### rendez-vous info ###########################
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_IP=$(getent hosts $MASTER_NODE | awk '{print $1}')
MASTER_PORT=$((49152 + RANDOM % 16384))

echo "MASTER ${MASTER_NODE} (${MASTER_IP}:${MASTER_PORT})"
echo "IMAGE  $IMAGE"
echo "CACHE  $HOST_CACHE → $CNTR_CACHE"

###################### helper #####################################
run_in_container () {
  local NODELIST=$1 ; shift
  srun --nodes=1 --ntasks=1 --exact --nodelist=$NODELIST \
       --container-image="$IMAGE" \
       --container-workdir=/app \
       --container-env="MASTER_ADDR=$MASTER_IP,MASTER_PORT=$MASTER_PORT" \
       --container-mounts="$HOST_CACHE:$CNTR_CACHE:rw" \
       "$@"
}

###################### 1) node-local GPU probe ####################
run_in_container $MASTER_NODE python -m gpu_probe.runner --test
PROBE_RC=$?

###################### 2) tiny CIFAR-10 run #######################
TRAIN_ARGS="--epochs 1 --batches_per_epoch 20 --lr 0.01 --data_path $CNTR_CACHE"
TORCHRUN_CMD="torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
  src/gpu_probe/train.py $TRAIN_ARGS"

# start rank-0 first so its TCP store is ready
run_in_container $MASTER_NODE bash -c "$TORCHRUN_CMD" &   # rank-0
sleep 25   # give the store time to bind
# launch the remaining node(s)
for NODE in $(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2); do
  run_in_container $NODE bash -c "$TORCHRUN_CMD" &
done
wait
TRAIN_RC=$?

###################### 3) final status ############################
[[ $PROBE_RC -eq 0 && $TRAIN_RC -eq 0 ]] && {
  echo "✅  All steps finished OK" ; exit 0 ; } || {
  echo "❌  probe=$PROBE_RC  train=$TRAIN_RC" ; exit 1 ; }