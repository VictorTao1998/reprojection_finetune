#!/bin/bash
export PYTHONWARNINGS="ignore"

python -m torch.distributed.launch /jianyu-fast-vol/ActiveZero/train_psmnet_sim_cv_adversarial.py \
--config-file '/jianyu-fast-vol/ActiveZero/configs/remote_train_primitive_randscenes.yaml' \
--logdir '/jianyu-fast-vol/eval/adv_deep/gp1' \
--model 'discriminator2' \
--b1 0.5 \
--b2 0.9 \
--discriminatorlr 0.0001 \
--clipc 0.001 \
--lam 10 \
--gaussian-blur \
--color-jitter \
--gradientpenalty