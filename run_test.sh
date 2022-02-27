#!/bin/bash
export PYTHONWARNINGS="ignore"

python /jianyu-fast-vol/ActiveZero/test_psmnet_sim.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/sim_cv_adv_2/train_sim_adv/models/model_100000.pth' \
--output '/jianyu-fast-vol/eval/sim_cv_adv_2/test_sim_adv' \
--exclude-bg \
--exclude-zeros 