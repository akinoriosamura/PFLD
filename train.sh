#!/usr/bin/env bash
num_labels=98
depth_multi=1 # default = 1, like model complicated
save_model=./models2/save_models/98/1111
file_list=data/train_original_data/list.txt
test_list=data/test_original_data/list.txt
pre_model=./models2/trained_models/WFLW_98/1004
lr=0.0001

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
nohup python -u train_model.py --model_dir=${save_model} \
                               --file_list=${file_list} \
                               --test_list=${test_list} \
                               --num_labels=${num_labels} \
                               --learning_rate=${lr} \
                               --level=L1 \
                               --debug=False \
                               --image_size=112 \
                               --batch_size=128 \
                               --depth_multi=${depth_multi} &
