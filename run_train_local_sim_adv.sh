#!/bin/bash
export PYTHONWARNINGS="ignore"
export CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.launch /code/ActiveZero/train_psmnet_sim_cv_adversarial.py \
--config-file '/code/ActiveZero/configs/local_train_primitive_steps.yaml' \
--logdir '/media/jianyu/dataset/eval/train_sim_cv_adversarial_depth/' \
--model 'discriminator1' \
--gaussian-blur \
--color-jitter \
--diffcv \
--kernel 3 \
--usedepth
#--b1 0.5 \
#--b2 0.9 \
#--discriminatorlr 0.00005 \
#--clipc 0.001 \
#--lam 10 \
#--gaussian-blur \
#--color-jitter \
#--diffcv \
#--kernel 7
#--gradientpenalty 