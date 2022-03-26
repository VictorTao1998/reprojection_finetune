#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/ActiveZero/train_psmnet_sim.py \
--config-file '/code/ActiveZero/configs/local_train_primitive_steps.yaml' \
--logdir '/media/jianyu/dataset/eval/train_psm_sim_depth/' \
--gaussian-blur \
--color-jitter \
--usedepth \
--summary-freq 50