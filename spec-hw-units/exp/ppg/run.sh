#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8

data_path="/space/risso/odimo_rebuttal/darkside_ppg"
save_path="/space/risso/odimo_rebuttal/darkside_ppg"

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

mkdir -p ${save_path}/${arch}_init_${init}_warmup_${wmup}
mkdir -p ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}

python main.py --arch ${arch} --checkpoint-dir ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost} \
               --data-dir ${data_path} ${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} \
               --strength ${strength} --seed ${seed} | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp}/log.txt

# python main.py --arch ${arch} --checkpoint-dir ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost} \
#                --data-dir ${data_path} ${timestamp} \
#                --world-size ${n_gpus} \
#                --epochs 400 --init-strategy ${init} \
#                --warmup --warmup-strategy ${wmup} \
#                --cost ${cost} ${ckp_search} \
#                --search --strength ${strength} --seed ${seed}

python main.py --arch ${arch} --checkpoint-dir ${save_path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost} \
               --data-dir ${data_path} ${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --cost ${cost} \
               --search --strength ${strength} --seed ${seed} \
               --finetune --finetune-scratch | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}_${cost}/${timestamp}/log.txt
