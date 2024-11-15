#!/bin/sh
dataset='mp3dhq0'
subset='f g h i j k l'
exp_name='eval_'
model_path='/home/NeuralCoMapping/train/models/infonce_num_robot_7/model_best.global'
for scene in ${subset}
do
    exp_name_full="${exp_name}_${dataset}${scene}"
    scene_path="scenes/${dataset}-${scene}.scenes"
    CUDA_VISIBLE_DEVICES=1 GIBSON_DEVICE_ID=1 nohup python main.py --exp_name ${exp_name_full} --scenes_file ${scene_path} --dump_location ./num7 --num_episodes 20 --load_global ${model_path} --num_local_steps 15 --num_global_steps 120 --num_robots 7
done
