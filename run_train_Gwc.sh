#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/reprojection_finetune/train_psmnet_temporal_ir_reproj_Gwc.py \
--config-file '/jianyu-fast-vol/reprojection_finetune/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/reprojection_gwc/train_reprojection_gwc_test_2/' \
--gaussian-blur \
--color-jitter