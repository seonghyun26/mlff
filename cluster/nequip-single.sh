#!/bin/sh

#SBATCH -J NequIP-noise
#SBATCH --out ./log/NequIP-noise-%j.out
#SBATCH -p A5000
## SBATCH --gres=gpu:8
## SBTACH --ntasks=8
## SBATCH --cpus-per-task=1
## SBATCH --tasks-per-node=8
## SBATCH --nodes=1

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBTACH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=0-72:00:00


cd $SLURM_SUBMIT_DIR
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONDA_ENVIRONMENT=$CONDA_DEFAULT_ENV"
# echo "CUDA_HOME=$CUDA_HOME"
# echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date
module purge

## 기타 실행할 스크립트를 여기 작성

date

nvidia-smi

cd ../scripts/train
./run_train_parallel.sh NequIP_debug

echo  "##### FINISHED #####"
