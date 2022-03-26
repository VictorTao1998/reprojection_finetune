#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch --nproc_per_node=2 /jianyu-fast-vol/ActiveZero/train_psmnet_sim.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/psm_depth/vanilla' \
--gaussian-blur \
--color-jitter \
--usedepth \
--summary-freq 100