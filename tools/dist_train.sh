#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

export TORCH_DISTRIBUTED_DEBUG='INFO'
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=2,3,4,5
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OMP_NUM_THREADS=6
export NCCL_P2P_LEVEL=NVL

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_UniBEV.py $CONFIG --launcher pytorch ${@:3}
