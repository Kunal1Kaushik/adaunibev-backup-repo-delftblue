#!/usr/bin/env bash

#SBATCH --job-name="create_nuscenes_annotations"
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-ME-cor
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_logs/create_nuscenes_annotations_a40_db_multi_node_%j.out
#SBATCH --error=outputs/slurm_logs/create_nuscenes_annotations_a40_db_multi_node_%j.err

module load 2024r1
module load openmpi
module load cudnn
module load nccl
module load cuda/12.5
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openmmlab


export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OMP_NUM_THREADS=16
export NCCL_P2P_LEVEL=NVL

set -x

#PARTITION=$1
#JOB_NAME=$2
#CONFIG='projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
#WORK_DIR='./work_dirs'
#GPUS=${GPUS:-8}
#GPUS_PER_NODE=${GPUS_PER_NODE:-8}
#CPUS_PER_TASK=${CPUS_PER_TASK:-5}
#SRUN_ARGS=${SRUN_ARGS:-""}
#PY_ARGS=${@:5}



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python -u tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes


conda deactivate

#srun -p ${PARTITION} \
#    --job-name=${JOB_NAME} \
#    --gres=gpu:${GPUS_PER_NODE} \
#    --ntasks=${GPUS} \
#    --ntasks-per-node=${GPUS_PER_NODE} \
#    --cpus-per-task=${CPUS_PER_TASK} \
#    --kill-on-bad-exit=1 \
#    ${SRUN_ARGS} \