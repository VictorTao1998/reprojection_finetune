#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/ActiveZero/train_psmnet_sim_cv_adversarial.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/sim_cv_adv_deep/train_sim_adv' \
--gaussian-blur \
--color-jitter