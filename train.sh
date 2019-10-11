#!/usr/bin/env bash
file_list=data/train_WFLW_68_data/list.txt
test_list=data/test_WFLW_68_data/list.txt
save_model=./models2/68/1010/save_model
pre_model=./models2/1004/pre_model
logs=./models2/log_1010.txt
num_labels=68
lr=0.000000001

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
                               > ${logs} 2>&1
tail -f ${logs}
 