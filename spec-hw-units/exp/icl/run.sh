#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8

path="."

strength=$1
arch=$2
init=$3
wmup=$4

if [[ "$5" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$5
fi

seed=$6

if [[ "$7" == "darkside" ]]; then
    cost="darkside"
elif [[ "$7" == "darkside-power" ]]; then
    cost="darkside-power"
else
    cost="naive"
fi

mkdir -p ${path}/${arch}_init_${init}_warmup_${wmup}
mkdir -p ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}
mkdir -p ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp}

python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} \
               --search --strength ${strength} --seed ${seed} | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp}/log.txt

python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} \
               --search --strength ${strength} --seed ${seed} \
               --finetune \
               --finetune-again \
               --finetune-scratch | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp}/log.txt

python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} \
               --search --strength ${strength} --seed ${seed} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp}/log.txt
