#!/usr/bin/env bash

path="/space/risso/odimo_rebuttal/diana"

strength=$1
timestamp=$2

python3 dig_after_pit_r20.py ${path}/model_${strength}/${timestamp} \
    --seed 42 | tee ${path}/model_${strength}/${timestamp}/log_quant.txt
