#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/train_psmnet_reprojection_real_finetune.py \
--config-file '/code/configs/local_train_primitive_steps.yaml' \
--logdir '/media/jianyu/dataset/eval/train_reprojection_finetune/' \
--model '/media/jianyu/dataset/eval/model_best_2.pth'