#!/usr/bin/env bash

strength=$1
path="/space/risso/odimo_rebuttal/diana"


if [[ "$2" == "now" ]]; then
    timestamp=$(date +"%Y-%m-%d-%T")
else
    timestamp=$4
fi

mkdir -p ${path}/model_${strength}
mkdir -p ${path}/model_${strength}/${timestamp}

python3 pit_r20.py ${path}/model_${strength}/${timestamp} \
    --seed 42 --strength ${strength} | tee ${path}/model_${strength}/${timestamp}/log_pit_${strength}.txt
