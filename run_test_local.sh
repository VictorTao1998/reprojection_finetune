#!/bin/bash
export PYTHONWARNINGS="ignore"

python /code/ActiveZero/test_sim_trans_local.py \
--config-file '/code/ActiveZero/configs/local_test.yaml' \
--model '/media/jianyu/dataset/eval/model_best_2.pth' \
--output '/media/jianyu/dataset/eval/cost_vol_visual/' \
--exclude-bg 