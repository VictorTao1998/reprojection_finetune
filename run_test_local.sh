#!/bin/bash
export PYTHONWARNINGS="ignore"

python /code/ActiveZero/test_sim_trans_local.py \
--config-file '/code/ActiveZero/configs/local_test.yaml' \
--model '/media/jianyu/dataset/eval/sim_vanilla/model_50000.pth' \
--output '/media/jianyu/dataset/eval/sim_test' \
--exclude-bg 