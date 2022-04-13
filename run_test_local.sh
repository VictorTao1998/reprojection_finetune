#!/bin/bash
export PYTHONWARNINGS="ignore"

python /code/ActiveZero/test_sim_local.py \
--config-file '/code/ActiveZero/configs/local_test.yaml' \
--model '/media/jianyu/dataset/eval/train_psm_sim_depth/models/model_50000.pth' \
--output '/media/jianyu/dataset/eval/psm_depth/vanilla' \
--exclude-bg \
--usedepth \
--cv