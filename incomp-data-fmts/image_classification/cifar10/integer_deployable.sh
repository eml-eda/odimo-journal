#!/usr/bin/env bash

strength=$1
arch=$2
path_srch=$3
path_ft=$4
path="."

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/model_${strength}
mkdir -p ${path}/${arch}/model_${strength}/deployment

python3 deploy.py ${path}/${arch}/model_${strenght}/deployment \
    -d cifar10 -a quant${arch} --ac ${path_srch} \
    --pretrained-w ${path_ft} \
    -b 128 --workers 0 --seed 42 --gpu 0 | tee ${path}/${arch}/model_${strenght}/deployment/log.txt
