#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/ActiveZero/test_psmnet_sim.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/sim_cv_bce_2/train_sim_cv_bce/models/model_100000.pth' \
--output '/jianyu-fast-vol/eval/sim_cv_bce_2/test_sim_cv_bce' \
--exclude-bg 