#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection_finetune/test_psmnet_reprojection.py \
--config-file '/jianyu-fast-vol/reprojection_finetune/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/best/model_best_2.pth' \
--output '/jianyu-fast-vol/eval/reprojection_best/' \
--onreal \
--exclude-bg 