#!/bin/bash

GPU=$1

CURRENT_PATH=$(pwd)
BENCHMARK_HOME=$(realpath ../../../)

cd $BENCHMARK_HOME

EXPDIR=/home/workspace/MLFF/NeurIPS23_DnB-exp/HfO_v1.0/dataset_2/GemNet-T

# OCP config
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/HfO_v1.0/gemnet-T.yml
#EXPID=Rmax6_MaxNeigh50_otf_NormOn_ReduceLROnPlateau_LR5e-3_EP80_E1_MAE_F100_L2MAE_EMA999_BS4_1V100

# with some SAIT modifications
EXPID=Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR5e-3_EP200_E1_MAE_F100_L2MAE_EMA999_BS4_1V100 



# paper model cofnig
CONFIG=/nas/SAIT-MLFF-Framework/NeurIPS23_DnB/configs/HfO_v1.0/paper_models/gemnet-T.yml
EXPID=Paper_Model_Rmax6_MaxNeigh50_otf_NormOn_LinearLR_LR5e-3_EP200_E1_MAE_F100_L2MAE_EMA999_BS8_1V100 


CUDA_VISIBLE_DEVICES=$GPU python /nas/SAIT-MLFF-Framework/main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $EXPDIR \
    --identifier $EXPID \
    --print-every 100 \
    --save-ckpt-every-epoch 10 \
    --checkpoint $2 

cd $CURRENT_PATH

