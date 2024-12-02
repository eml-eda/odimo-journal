#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8

path="/space/risso/odimo_rebuttal/darkside_ppg"

arch=$1

if [[ "$2" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$2
fi

seed=$3

mkdir -p ${path}/${arch}
mkdir -p ${path}/${arch}/${timestamp}

python baseline.py --arch ${arch} --checkpoint-dir ${path}/${arch}/${timestamp} \
                   --data-dir ${path} \
                   --epochs 500 --seed ${seed} \
                   | tee -a ${path}/${arch}/${timestamp}/log.txt
