
## BUILD

```shell
conda create -p ./venv python=3.6
source activate ./venv
sh ./build.sh && python -m gibson2.utils.assets_utils --download_assets
```



## DATASET

+ Gibson

1. get dataset [here](https://forms.gle/36TW9uVpjrE1Mkf9A)

2. copy URL of `gibson_v2_4+.tar.gz`

3. run command

  ```shell
  python -m gibson2.utils.assets_utils --download_dataset {URL}
  ```


+ Matterport3D

1. get dataset according to [README](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md)

2. run command

  ```shell
  python2 download_mp.py --task_data igibson -o . `
  ```

3. move each folder of scenes to `Gibson Dataset path`

  You can check `Gibson Dataset path` by running

  ```shell
  python -m gibson2.utils.assets_utils --download_assets
  ```



## USAGE

+ Train

```shell
python main.py --global_lr 5e-4 --exp_name 'AIM_del_mi' --critic_lr_coef 5e-2 --train_global 1 --dump_location train --scenes_file scenes/train.scenes
```

+ Test (Example)

```shell
python main.py --exp_name 'eval_coscan_mp3dhq0f' --scenes_file scenes/mp3dhq0-f.scenes --dump_location std --num_episodes 10 --load_global best.global
```


+ Analyze performance

```shell
python analyze.py --dir std --dataset gibson -ne 5 --bins 35,70
python analyze.py --dir std --dataset mp3d -ne 5 --bins 100
```

+ Analyze performance

```shell
python scripts/easy_analyze.py rl --dataset hq --subset abcdef --dir std
```

+ Specify GPU Index

```shell
export CUDA_VISIBLE_DEVICES=0
export GIBSON_DEVICE_ID=0
```

+ Visualization

```shell
python main.py --exp_name 'eval_-72' --scenes_file scenes/mp3dhq0-f.scenes --dump_location ./temp --num_episodes 1 --load_global ./model_best.global  --vis_type 2
# dump at ./video/
python scripts/map2d.py  --exp_name 'eval_new-72' -ne 1 -ns 4
```
