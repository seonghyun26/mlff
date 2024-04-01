#!/bin/sh

#SBATCH -J NequIP-noise
#SBATCH --out ./log/NequIP-noise-%j.out
#SBATCH -p 3090
##SBATCH -p A100-40GB
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBTACH --ntasks=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=2
##SBATCH --nodelist=n8

cd $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# echo "CUDA_HOME=$CUDA_HOME"
# echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge

## 기타 실행할 스크립트를 여기 작성

date
nvidia-smi

CONDA_ENV="mlff"
# conda activate $CONDA_ENV

cd ../scripts/train
./run_train_parallel.sh NequIP_noise

echo  "##### END #####"
