#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/ActiveZero/train_psmnet_sim_ray_overfit.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/psm_ray/' \
--gaussian-blur \
--color-jitter \
--summary-freq 50 \
--debug