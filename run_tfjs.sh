#!/bin/bash
phase=$1
num_labels=68 # 20, 52
depth_multi=0.75 # default = 1, like model complicated
num_quant=64
save_model=models2/save_models/68/tfjs_aug_clev2_grow_helen_wflw
file_list=/data/pcn_aug_clev2_grow_helen_wflw_68/rotated_train_aug_clev2_grow_helen_wflw_68/list.txt
test_list=/data/pcn_aug_clev2_grow_helen_wflw_68/rotated_test_aug_clev2_grow_helen_wflw_68/list.txt
# test_list=data/prepro_dataset/test_moru_dataset/list.txt
pre_model=models2/save_models/68/tfjs
out_dir=sam_tfjs
lr=0.0001
is_augment=False  # True or False
image_size=112

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_hlo_profile

if [ ${phase} = "pretrain" ]; then
    echo "run pretrain"
    python -u train_model_tfjs.py --model_dir=${save_model} \
                                --pretrained_model=${pre_model} \
                                --file_list=${file_list} \
                                --test_list=${test_list} \
                                --num_labels=${num_labels} \
                                --learning_rate=${lr} \
                                --level=L1 \
                                --debug=False \
                                --image_size=${image_size} \
                                --batch_size=128 \
                                --depth_multi=${depth_multi} \
                                --num_quant=${num_quant} \
                                --is_augment=${is_augment}
elif [ ${phase} = "train" ]; then
    echo "run train"
    python -u train_model_tfjs.py --model_dir=${save_model} \
                                --file_list=${file_list} \
                                --test_list=${test_list} \
                                --num_labels=${num_labels} \
                                --learning_rate=${lr} \
                                --level=L1 \
                                --debug=False \
                                --image_size=${image_size} \
                                --batch_size=128 \
                                --depth_multi=${depth_multi} \
                                --num_quant=${num_quant} \
                                --is_augment=${is_augment}
elif [ ${phase} = "save" ]; then
    echo "run save"
    python -u save_tfjs_model.py --model_dir=${pre_model} \
                            --pretrained_model=${pre_model} \
                            --test_list=${test_list} \
                            --num_labels=${num_labels} \
                            --learning_rate=${lr} \
                            --level=L1 \
                            --image_size=${image_size} \
                            --batch_size=128 \
                            --depth_multi=${depth_multi} \
                            --num_quant=${num_quant} \
                            --is_augment=${is_augment}

elif [ "${phase}" = 'test' ]; then
    echo "run test"
    python -u test_tfjs_model.py --pretrained_model=${pre_model} \
                            --test_list=${test_list} \
                            --num_labels=${num_labels} \
                            --learning_rate=${lr} \
                            --level=L1 \
                            --image_size=${image_size} \
                            --batch_size=128 \
                            --depth_multi=${depth_multi} \
                            --out_dir=${out_dir} \
                            --num_quant=${num_quant} \
                            --is_augment=${is_augment}
else
    echo "no running"
fi
