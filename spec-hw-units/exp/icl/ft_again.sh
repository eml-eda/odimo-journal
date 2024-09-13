#!/bin/bash

path="."
arch="mbv1_search_8"
init="half"
wmup="fine"

strength="0.0e+0"
timestamp="2023-05-23-20:25:35"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="5.0e-10"
timestamp="2023-05-25-10:38:28"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

# strength="5.0e-9"
# timestamp="2023-05-25-14:43:06"
# python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
#                --epochs 500 --init-strategy ${init} \
#                --warmup --warmup-strategy ${wmup} \
#                --search --strength ${strength} \
#                --finetune \
#                --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="5.0e-8"
timestamp="2023-05-24-08:07:06"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="5.0e-7"
timestamp="2023-05-24-10:59:45"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="1.0e-6"
timestamp="2023-05-24-14:40:14"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="5.0e-6"
timestamp="2023-05-24-18:11:00"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="1.0e-5"
timestamp="2023-05-24-20:55:33"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="5.0e-5"
timestamp="2023-05-24-23:32:03"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="7.0e-5"
timestamp="2023-05-25-18:22:29"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="8.0e-5"
timestamp="2023-05-25-20:50:24"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="1.0e-4"
timestamp="2023-05-25-01:54:22"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt

strength="5.0e-4"
timestamp="2023-05-25-04:44:55"
python main.py --arch ${arch} --checkpoint-dir ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp} \
               --epochs 500 --init-strategy ${init} \
               --warmup --warmup-strategy ${wmup} \
               --search --strength ${strength} \
               --finetune \
               --finetune-again | tee -a ${path}/${arch}_init_${init}_warmup_${wmup}/model_${strength}/${timestamp}/log.txt