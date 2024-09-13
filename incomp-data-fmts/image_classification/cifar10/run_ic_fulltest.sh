#!/usr/bin/env bash

strength=$1
path="."

#arch="res8_fp"
#arch="res8_w8a8"
#arch="res8_w4a8"
#arch="res8_w2a8"
#arch="res8_w248a8_chan"
#arch="res8_w248a8_multiprec"
#arch="res8_w248a248_multiprec"
pretrained_model="warmup20_fp.pth.tar"
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

if [[ "$5" == "search" ]]; then
    echo Search
    split=0.0
    # NB: add --warmup-8bit if needed
    python3 search_r20.py ${path}/${arch}/model_${strength}/${timestamp} -a mix${arch} \
        -d cifar10 --arch-data-split ${split} \
        --epochs 200 --step-epoch 50 -b 128 -j 4 \
        --ac ${pretrained_model} --patience 50 \
        --lr 0.001 --lra 0.001 --wd 1e-4 \
        --ai same --cd ${strength} --target ${target} \
        --seed 42 --gpu 0 \
        --no-gumbel-softmax --temperature 1 --anneal-temp \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_search_${strength}.txt
fi

if [[ "$6" == "ft" ]]; then
    echo Fine-Tune
    python3 main_r20.py ${path}/${arch}/model_${strength}/${timestamp} -a quant${arch} \
        -d cifar10 --epochs 200 --step-epoch 50 -b 128 --patience 500 \
        --lr 0.0001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${arch}/model_${strength}/${timestamp}/arch_model_best.pth.tar -ft \
        --visualization -pr ${project} --tags ${tags} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_finetune_${strength}.txt
else
    echo From-Scratch
    #pretrained_model="${arch}/model_${strength}/arch_model_best.pth.tar"
    #pretrained_model="warmup_8bit.pth.tar"
    #pretrained_model="warmup_5bit.pth.tar"
    pretrained_model="warmup20_fp.pth.tar"
    python3 main_r20.py ${path}/${arch}/model_${strength}/${timestamp} -a quant${arch} \
        -d cifar10 --epochs 200 --step-epoch 50 -b 128 --patience 50 \
        --lr 0.001 --wd 1e-4 \
        --seed 42 --gpu 0 \
        --ac ${pretrained_model} | tee ${path}/${arch}/model_${strength}/${timestamp}/log_fromscratch_${strength}.txt
fi