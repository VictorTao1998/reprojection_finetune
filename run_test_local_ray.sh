#!/bin/bash
export PYTHONWARNINGS="ignore"

python /code/ActiveZero/test_sim_local_ray.py \
--config-file '/code/ActiveZero/configs/local_test.yaml' \
--model '/media/jianyu/dataset/eval/train_psm_sim_ray/models/model_50000.pth' \
--output '/media/jianyu/dataset/eval/psm_ray/' \
--exclude-bg 