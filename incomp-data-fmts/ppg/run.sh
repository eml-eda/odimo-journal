#!/usr/bin/env bash

strength=$1
path="/space/risso/odimo_rebuttal/diana_ppg"

# pretrained_model=""
arch=$2
target=$3

project="hp-nas_ic"

#tags="warmup"
#tags="fp"
#tags="init_same no_wp reg_w"
tags="init_same wp reg_w softemp"

if [[ "$4" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$4
fi

# timestamp=$(date +"%Y-%m-%d-%T")
mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}
mkdir -p ${path}/${arch}/model_${strength}/${timestamp}

export WANDB_MODE=offline

subject=9
pretrained_model="warmup_fp_s${subject}.pth.tar"

if [[ "$5" == "search" ]]; then
    echo Search
    split=0.0
    # NB: add --warmup-8bit if needed
    python3 search.py ${path}/${arch}/model_${strength}/${timestamp} -a mix${arch} \
        -d dalia --arch-data-split ${split} \
        --epochs 500 --step-epoch 50 -b 128 -j 4 \
        --ac ${pretrained_model} --patience 20 \
        --lr 0.001 --lra 0.0005 --wd 1e-4 \
        --ai same --cd ${strength} --target ${target} \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 1 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} \
        --subject ${subject} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_search_${strength}.txt
fi

if [[ "$6" == "ft" ]]; then
    echo Fine-Tune
    python3 main.py ${path}/${arch}/model_${strength}/${timestamp} -a quant${arch} \
        -d dalia --epochs 500 --step-epoch 50 -b 128 --patience 20 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${path}/${arch}/model_${strength}/${timestamp}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} \
        --subject ${subject} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_finetune_${strength}.txt
else
    echo From-Scratch
    # pretrained_model="warmup_fp.pth.tar"
    # pretrained_model="."
    python3 main.py ${path}/${arch}/model_${strength}/${timestamp} -a quant${arch} \
        -d dalia --epochs 500 --step-epoch 50 -b 128 --patience 20 \
        --lr 0.001 \
        --seed 42 --gpu 0 \
        --ac ${pretrained_model} \
        --subject ${subject} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_fromscratch_${strength}.txt
fi