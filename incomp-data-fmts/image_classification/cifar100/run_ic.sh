#!/usr/bin/env bash

WORLD_SIZE=1
VAL_SPLIT=0.1
ARCH_CFG="./resnet18_fp.pt"

strength=$1
data="/space/risso/cifar-100"
arch=$2
target=$3

if [[ "$4" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$4
fi
# mkdir -p ${arch}
# mkdir -p ${arch}/model_${strength}
# mkdir -p ${arch}/model_${strength}/${timestamp}

if [[ "$5" == "search" ]]; then
    echo Search
    split=0.0
    # NB: add --warmup-8bit if needed
    python3 search.py ${data} -a mix${arch} \
        --val-split 0.1 \
        --epochs 200 -b 128 \
        --patience 100 --timestamp ${timestamp} \
        --lr 0.001 --lra 0.001 --wd 5e-4 \
        --ai same --cd ${strength} --target ${target} \
        --seed 42 --workers 4 --world-size ${WORLD_SIZE} \
        --no-gumbel-softmax --temperature 1 --anneal-temp \
        --arch-cfg ${ARCH_CFG}
fi

# TODO: Undestand proper value of LR
if [[ "$6" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${data} -a quant${arch} \
        --epochs 200 -b 128 --patience 100 --timestamp ${timestamp} \
        --lr 0.001 --wd 5e-4  --cd ${strength} \
        --seed 42 --workers 4 --world-size ${WORLD_SIZE} \
        --ac ${arch}/model_${strength}/${timestamp}/best.pt -ft
else
    echo From-Scratch

    python3 main.py ${data} -a quant${arch} \
        --val-split ${VAL_SPLIT} --timestamp ${timestamp} \
        --epochs 200 --step-epoch 10 -b 128 --patience 100 \
        --lr 0.001 --wd 5e-4 \
        --seed 42 --workers 4 --world-size ${WORLD_SIZE} \
        --arch-cfg ${ARCH_CFG}
fi
