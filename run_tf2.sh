#!/bin/bash
phase=$1
num_labels=68 # 20, 52
depth_multi=1 # default = 1, like model complicated
num_quant=64
save_model=models2/save_models/68/sam_pfldtf2_300W_simple_wing
file_list=data/rotated_train_300W/list.txt
test_list=data/rotated_test_300W/list.txt
# file_list=data/test_moru_dataset/list.txt
# test_list=data/test_moru_dataset/list.txt
pre_model=models2/save_models/68/sam_pfldtf2_300W_simple_wing
out_dir=sam_pfldtf2_300W_simple_wing
lr=0.0001
is_augment=False  # True or False
image_size=112
batch_size=16

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_hlo_profile

if [ ${phase} = "pretrain" ]; then
    echo "run pretrain"
    python -u train_model_tf2.py --model_dir=${save_model} \
                                --pretrained_model=${pre_model} \
                                --file_list=${file_list} \
                                --test_list=${test_list} \
                                --num_labels=${num_labels} \
                                --learning_rate=${lr} \
                                --level=L1 \
                                --debug=False \
                                --image_size=${image_size} \
                                --batch_size=${batch_size} \
                                --depth_multi=${depth_multi} \
                                --num_quant=${num_quant} \
                                --is_augment=${is_augment}
elif [ ${phase} = "train" ]; then
    echo "run train"
    python -u train_model_tf2.py --model_dir=${save_model} \
                                --file_list=${file_list} \
                                --test_list=${test_list} \
                                --num_labels=${num_labels} \
                                --learning_rate=${lr} \
                                --level=L1 \
                                --debug=False \
                                --image_size=${image_size} \
                                --batch_size=${batch_size} \
                                --depth_multi=${depth_multi} \
                                --num_quant=${num_quant} \
                                --is_augment=${is_augment}
elif [ ${phase} = "save" ]; then
    echo "run save"
    python -u save_model.py --model_dir=${pre_model} \
                            --pretrained_model=${pre_model} \
                            --test_list=${test_list} \
                            --num_labels=${num_labels} \
                            --learning_rate=${lr} \
                            --level=L1 \
                            --image_size=${image_size} \
                            --batch_size=${batch_size} \
                            --depth_multi=${depth_multi} \
                            --num_quant=${num_quant} \
                            --is_augment=${is_augment}

elif [ "${phase}" = 'test' ]; then
    echo "run test"
    python -u test_model_tf2.py --pretrained_model=${pre_model} \
                            --file_list=${file_list} \
                            --test_list=${test_list} \
                            --num_labels=${num_labels} \
                            --learning_rate=${lr} \
                            --level=L1 \
                            --image_size=${image_size} \
                            --batch_size=${batch_size} \
                            --depth_multi=${depth_multi} \
                            --out_dir=${out_dir} \
                            --num_quant=${num_quant} \
                            --is_augment=${is_augment}
else
    echo "no running"
fi
