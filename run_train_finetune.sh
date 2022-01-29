#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection_finetune/train_psmnet_temporal_ir_finetune.py \
--config-file '/jianyu-fast-vol/reprojection_finetune/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/reprojection_finetune/train_reprojection_finetune/' \
--model '/jianyu-fast-vol/eval/reprojection_finetune/train_reprojection_sim/models/model_40000.pth' \
--gaussian-blur \
--color-jitter