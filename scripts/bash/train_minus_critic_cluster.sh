#!/bin/sh
exp_name='Asymmetric_NLMinus_AC_Cluster_PreLoss'
nohup python main.py --global_lr 5e-5 --exp_name ${exp_name} --critic_lr_coef 5e-1 --train_global 1 --dump_location train --scenes_file scenes/train.scenes --num_episodes 2500 --num_global_steps 72 --predict_loss_coef 0.05