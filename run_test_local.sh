#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /code/test_psmnet_reprojection.py \
--config-file '/code/configs/local_test_on_real.yaml' \
--model '/media/jianyu/dataset/eval/train_reprojection_finetune/models/model_40000.pth' \
--output '/media/jianyu/dataset/eval/finetune_test_data' \
--exclude-bg 