#!/bin/bash
export PYTHONWARNINGS="ignore"

python /code/ActiveZero/test_sim_local.py \
--config-file '/code/ActiveZero/configs/local_test.yaml' \
--model '/media/jianyu/dataset/eval/adv_kernel/model_73000.pth' \
--output '/media/jianyu/dataset/eval/adv_kernel/' \
--exclude-bg 