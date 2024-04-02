#!/bin/bash

GPU=$1
# CKPT_DIR=$(realpath "../../train_results/HfO/NequIP_test/checkpoints/")
# LATEST_CKPT_DIR=$(find $CKPT_DIR -name "checkpoint.pt" -exec dirname {} \; | sort -nr | head -n 1)
CKPT_DIR=$(realpath "../../train_results/HfO/NequIP_noise/checkpoints/train-20240401_051952/")
DATA_LMDB=$(realpath "../../datasets/HfO/atom_graph_rmax6.0_maxneighbor50/test.lmdb")
echo ${CKPLATEST_CKPT_DIRT_DIR}

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

ROUND_NUMBER=4

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/ckpt_round${ROUND_NUMBER}.pt \
    --validate-data $DATA_LMDB \
