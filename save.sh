#!/usr/bin/env bash
num_labels=98
depth_multi=1 # default = 1, like model complicated
save_model=./models2/save_models/98/1111
pre_model=./models2/save_models/98/1111
lr=0.000000001

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export XLA_FLAGS=--xla_hlo_profile
python -u save_model.py --model_dir=${save_model} \
                         --pretrained_model=${pre_model} \
                         --num_labels=${num_labels} \
                         --learning_rate=${lr} \
                         --level=L1 \
                         --debug=False \
                         --image_size=112 \
                         --batch_size=128 \
                         --depth_multi=${depth_multi}
