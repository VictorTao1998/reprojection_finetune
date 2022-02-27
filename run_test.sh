#!/bin/bash
export PYTHONWARNINGS="ignore"

python /jianyu-fast-vol/ActiveZero/test_psmnet_sim.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/sim_cv_bce/train_sim/models/model_100000.pth' \
--output '/jianyu-fast-vol/eval/sim_psm/test_sim_n' \
--exclude-bg \
--exclude-zeros 