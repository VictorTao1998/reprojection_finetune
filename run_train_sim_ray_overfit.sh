#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch --nproc_per_node=2 /jianyu-fast-vol/ActiveZero/train_psmnet_sim_ray_overfit.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/psm_ray/' \
--summary-freq 50 \
--n_rays 1024