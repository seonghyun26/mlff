#!/bin/bash

GPU=$1
# CKPT_DIR=$(realpath "../../train_results/HfO/NequIP_test/checkpoints/")
# LATEST_CKPT_DIR=$(find $CKPT_DIR -name "checkpoint.pt" -exec dirname {} \; | sort -nr | head -n 1)
CKPT_DIR=$(realpath "../../train_results/HfO/NequIP_test/checkpoints/train-20240315_040043/")
DATA_LMDB=$(realpath "../../datasets/HfO/ood/atom_graph_rmax6.0_maxneighbor50/ood.lmdb")
echo ${CKPLATEST_CKPT_DIRT_DIR}

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode validate \
    --config-yml ${CKPT_DIR}/config_train.yml \
    --checkpoint ${CKPT_DIR}/checkpoint.pt \
    --validate-data $DATA_LMDB \
