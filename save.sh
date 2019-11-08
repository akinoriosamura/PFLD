#!/usr/bin/env bash
num_labels=98
save_model=./models2/save_models/98/1107
pre_model=./models2/trained_models/WFLW_98/1004
lr=0.000000001

# --pretrained_model=${pre_model} \
# CUDA_VISIBLE_DEVICES='' \
python -u train_model.py --model_dir=${save_model} \
                         --pretrained_model=${pre_model} \
                         --num_labels=${num_labels} \
                         --learning_rate=${lr} \
                         --level=L1 \
                         --debug=False \
                         --image_size=112 \
                         --batch_size=128
 
# #!/usr/bin/env bash
# num_labels=68
# save_model=./models2/save_models/68/1101
# pre_model=./models2/trained_models/WFLW_98/1004
# logs=./models2/log_1101.txt
# 
# # --pretrained_model=${pre_model} \
# # CUDA_VISIBLE_DEVICES='' \
# python -u train_model.py --model_dir=${save_model} \
#                                --pretrained_model=${pre_model} \
#                                --num_labels=${num_labels} \
#                                --learning_rate=${lr} \
#                                --level=L1 \
#                                --debug=False \
#                                --image_size=112 \
#                                --batch_size=128
