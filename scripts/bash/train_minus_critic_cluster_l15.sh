#!/bin/sh
exp_name='Asymmetric_PreLoss_l15_fixcluster'
nohup python main.py --global_lr 5e-5 --exp_name infonce --critic_lr_coef 5e-1 --train_global 1 --dump_location train --scenes_file scenes/train.scenes --num_episodes 2500 --predict_loss_coef 0.05 --num_local_steps 15 --num_global_steps 120