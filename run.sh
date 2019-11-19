#!/bin/bash
phase=$1
num_labels=68
depth_multi=1 # default = 1, like model complicated
save_model=models2/save_models/quant_growing_pre_WFLW
file_list=data/train_growing_data/list.txt
test_list=data/test_growing_data/list.txt
pre_model=models2/save_models/growing_pre_WFLW
out_dir=sample_result
lr=0.0001

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_hlo_profile

if [ ${phase} = "pretrain" ]; then
    echo "run pretrain"
    python -u train_model.py --model_dir=${save_model} \
                                --pretrained_model=${pre_model} \
                                --file_list=${file_list} \
                                --test_list=${test_list} \
                                --num_labels=${num_labels} \
                                --learning_rate=${lr} \
                                --level=L1 \
                                --debug=False \
                                --image_size=112 \
                                --batch_size=128 \
                                --depth_multi=${depth_multi}
elif [ ${phase} = "train" ]; then
    echo "run train"
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
elif [ ${phase} = "save" ]; then
    echo "run save"
    python -u save_model.py --model_dir=${save_model} \
                            --pretrained_model=${pre_model} \
                            --num_labels=${num_labels} \
                            --learning_rate=${lr} \
                            --level=L1 \
                            --image_size=112 \
                            --batch_size=128 \
                            --depth_multi=${depth_multi}

elif [ "${phase}" = 'test' ]; then
    echo "run test"
    python -u test_model.py --pretrained_model=${pre_model} \
                            --test_list=${test_list} \
                            --num_labels=${num_labels} \
                            --learning_rate=${lr} \
                            --level=L1 \
                            --image_size=112 \
                            --batch_size=128 \
                            --depth_multi=${depth_multi} \
                            --out_dir=${out_dir}
else
    echo "no running"
fi