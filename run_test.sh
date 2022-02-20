#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection_finetune/test_psmnet_sim.py \
--config-file '/jianyu-fast-vol/reprojection_finetune/configs/remote_test.yaml' \
--model '/jianyu-fast-vol/eval/sim_cv_bce/train_sim_cv_bce/models/model_40000.pth' \
--output '/jianyu-fast-vol/eval/sim_cv_bce/test_sim_cv_bce' \
--exclude-bg 