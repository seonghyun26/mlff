#!/bin/bash

GPU=$1
# GPU=6

MODEL_ARRAY=(
    "BPNN"
    "SchNet"
    "DimeNet++"
    "GemNet-T"
    "GemNet-dT"
    "NequIP"
    "Allegro"
    "MACE"
    "SCN"
)
# MODEL=$2
MODEL=NequIP_test

DATA_ARRAY=(
    "SiN"
    "HfO"
)
# DATA=$3
DATA=HfO

BENCHMARK_HOME=$(realpath ../../)
cd $BENCHMARK_HOME

CONFIG=configs/train/${DATA}/${MODEL}.yml
RUNDIR=train_results/${DATA}/${MODEL}
RUNID=train

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --mode train \
    --config-yml $CONFIG \
    --run-dir $RUNDIR \
    --identifier $RUNID \
    --print-every 100 \
    --save-ckpt-every-epoch 20 \


