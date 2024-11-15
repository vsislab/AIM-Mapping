#!/bin/sh
dataset='mp3dhq0'
subset='a b c d e f g h i j k l'
exp_name='eval_Asymmetric_NLMinus_AC_Cluster_LowLr'
model_path='train/dump/Asymmetric_NLMinus_AC_Cluster_LowLr/periodic_89500008.global'
for scene in ${subset}
do
    exp_name_full="${exp_name}_${dataset}${scene}"
    scene_path="scenes/${dataset}-${scene}.scenes"
    CUDA_VISIBLE_DEVICES=0 python main.py --exp_name ${exp_name_full} --scenes_file ${scene_path} --dump_location ./temp --num_episodes 50 --load_global ${model_path} 
done
