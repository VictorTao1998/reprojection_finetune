#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection_finetune/test_psmnet_reprojection_Gwc.py \
--config-file '/jianyu-fast-vol/reprojection_finetune/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/reprojection_finetune/train_reprojection_Gwc/models/model_36000.pth' \
--output '/jianyu-fast-vol/eval/reprojection_finetune/test_reprojection_Gwc/' \
--onreal \
--exclude-bg 