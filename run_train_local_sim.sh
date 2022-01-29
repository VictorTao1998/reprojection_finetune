#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/train_psmnet_reprojection_sim.py \
--config-file '/code/configs/local_train_primitive_steps.yaml' \
--logdir '/media/jianyu/dataset/eval/train_reprojection_sim/' 