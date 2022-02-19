#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection_finetune/train_psmnet_sim_cv_bce.py \
--config-file '/jianyu-fast-vol/reprojection_finetune/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/sim_cv_bce/train_sim_cv_bce' \
--gaussian-blur \
--color-jitter