#!/usr/bin/env bash

strength=$1
data="/space/risso/imagenet-1k/imagenet"
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
        --epochs 130 -b 128 \
        --patience 25 --timestamp ${timestamp} \
        --lr 0.001 --lra 0.001 --wd 1e-4 \
        --ai same --cd ${strength} --target ${target} \
        --seed 42 --workers 4 --world-size 2 \
        --no-gumbel-softmax --temperature 1 --anneal-temp
fi

if [[ "$6" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${data} -a quant${arch} \
        --epochs 130 -b 128 --patience 50 --timestamp ${timestamp} \
        --lr 0.0005 --wd 1e-4  --cd ${strength} \
        --seed 42 --workers 4 --world-size 2 --step-size 30 \
        --ac ${arch}/model_${strength}/${timestamp}/best.pt -ft
else
    echo From-Scratch
    python3 main.py ${data} -a quant${arch} \
        --epochs 130 -b 128 --patience 50 --timestamp ${timestamp} \
        --lr 0.01 --wd 1e-4 --cd ${strength} \
        --seed 42 --workers 4 --world-size 2 --step-size 30
fi
