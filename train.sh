#!/usr/bin/env bash
num_labels=68
depth_multi=1 # default = 1, like model complicated
save_model=./models2/save_models/pcn_WFLW_68/1113
file_list=data/non_rotated_train_pcn_WFLW_68_data/list.txt
test_list=data/non_rotated_test_pcn_WFLW_68_data/list.txt
pre_model=models2/trained_models/WFLW_68/PFLD68WFLW_models_1112
lr=0.0001

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
python -u train_model.py --model_dir=${save_model} \
                               --file_list=${file_list} \
                               --test_list=${test_list} \
                               --num_labels=${num_labels} \
                               --learning_rate=${lr} \
                               --level=L1 \
                               --debug=False \
                               --image_size=112 \
                               --batch_size=128 \
                               --depth_multi=${depth_multi}
