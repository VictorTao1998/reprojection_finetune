#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/ActiveZero/train_psmnet_sim_cv_bce.py \
--config-file '/code/ActiveZero/configs/local_train_primitive_steps.yaml' \
--logdir '/media/jianyu/dataset/eval/train_reprojection_sim_cv_bce/' \
--gaussian-blur \
--color-jitter