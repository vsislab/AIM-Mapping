#!/bin/sh
dataset='train'
subset='a'
exp_name='eval_Asymmetric_PreLoss_l15_fixcluster'
model_path='train/dump/Asymmetric_PreLoss_l15_fixcluster/periodic_8000004.global'
for scene in ${subset}
do
    exp_name_full="${exp_name}_${dataset}${scene}"
    scene_path="scenes/${dataset}-${scene}.scenes"
    CUDA_VISIBLE_DEVICES=1 GIBSON_DEVICE_ID=1 nohup python main.py --seed 11 --exp_name ${exp_name_full} --scenes_file ${scene_path} --dump_location ./temp --num_episodes 50 --load_global ${model_path} --num_local_steps 15 --num_global_steps 200 
done
