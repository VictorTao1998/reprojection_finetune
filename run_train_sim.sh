#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/ActiveZero/train_psmnet_sim_cv_bce.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/loss_function/sim_bce' \
--gaussian-blur \
--color-jitter