#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8

data_path="/space/risso/imagenet-1k/imagenet"
save_path="/space/risso/odimo/imagenet"

strength=$1
arch=$2
init=$3
wmup=$4

if [[ "$5" == "now" ]]; then
    timestamp=""
else
    timestamp="--timestamp $5"
fi

seed=$6

if [[ "$7" == "darkside" ]]; then
    cost="darkside"
else
    cost="naive"
fi

n_gpus=$8

if [[ "$9" == "no_ckp_wup" ]]; then
    ckp_wup=""
else
    ckp_wup="--resume-ckp-warmup $9"
fi

if [[ "${10}" == "no_ckp_search" ]]; then
    ckp_search=""
else
    ckp_search="--resume-ckp-search ${10}"
fi

mkdir -p ${save_path}/${arch}_init_${init}_warmup_${wmup}
mkdir -p ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}

python main.py --arch ${arch} --checkpoint-dir ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost} \
               --data-dir ${data_path} ${timestamp} \
               --use-ema --world-size ${n_gpus} \
               --epochs 300 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} ${ckp_wup} \
               --strength ${strength} --seed ${seed}

python main.py --arch ${arch} --checkpoint-dir ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost} \
               --data-dir ${data_path} ${timestamp} \
               --world-size ${n_gpus} \
               --epochs 300 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} ${ckp_search} \
               --search --strength ${strength} --seed ${seed}
