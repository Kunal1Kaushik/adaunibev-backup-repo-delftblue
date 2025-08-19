#!/usr/bin/env bash

#SBATCH --job-name="test_unibev_40000weights"
#SBATCH --partition=gpu
#SBATCH --time=1:30:00

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-ME-cor
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_logs/test_unibev_40000weights_a40_db_multi_node_%j.out
#SBATCH --error=outputs/slurm_logs/test_unibev_40000weights_a40_db_multi_node_%j.err

#module load 2024r1
#module load openmpi
#module load cudnn
#module load nccl
#module load cuda/12.5
#module load miniconda3

source ~/.bashrc

#unset CONDA_SHLVL
#source "$(conda info --base)/etc/profile.d/conda.sh"

# Disable hostname resolution (critical for Singularity + Slurm)
export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=29500

srun singularity exec --nv \
  --bind /projects:/projects \
  unibev_cuda.sif \
  bash -c '
    source ~/.bashrc_cuda113

    
    conda activate unibev

    cd mmdetection3d/

    export MKL_NUM_THREADS=16
    export NUMEXPR_NUM_THREADS=16
    export OMP_NUM_THREADS=16
    export NCCL_P2P_LEVEL=NVL
    export CUDA_LAUNCH_BLOCKING=1

    set -x

    #PARTITION=$1
    #JOB_NAME=$2
    CONFIG="projects/UniBEV/configs/unibev/unibev_nus_LC_avg_256_modality_dropout_customtraining.py"
    CHECKPOINT="./unibev_2500weights_severitysquared_fovlostandbeamsmiss_correct/epoch_7.pth"
    #GPUS=${GPUS:-8}
    #GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    #CPUS_PER_TASK=${CPUS_PER_TASK:-5}
    #SRUN_ARGS=${SRUN_ARGS:-""}
    #PY_ARGS=${@:5}

    echo "Current Working Directory: $(pwd)"

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -u tools/test_UniBEV.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --eval mAP NDS ${PY_ARGS}

    conda deactivate
'