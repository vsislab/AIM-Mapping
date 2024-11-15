import matplotlib.pyplot as plt
import argparse
import os
import re
import numpy as np


def main(path,path2):
    timestep2 = 0
    critic_losses2 = []
    actor_losses2 = []
    steps2 = []
    mean_reward2 = []
    eps_step2 = 0
    ppo_sample_eps2 = 0
    exp_name2 = 'exp'
    eps2 = []
    with open(path2, 'r') as f:
        for line2 in f.readlines():
            if len(line2) < 5:
                continue
            line2 = line2.rstrip()
            if 'Namespace(' in line2:
                res2 = re.match(r'.*?exp_name=\'(.*?)\'.*?max_episode_length=([0-9]*).*?num_processes=([0-9]*).*?ppo_sample_eps=([0-9]*).*', line2)
                exp_name2 = res2.group(1)
                eps_step2 = int(res2.group(2)) * int(res2.group(3))
                ppo_sample_eps2 = int(res2.group(4))
                eps2 = []
                critic_losses2 = []
                actor_losses2 = []
                value_losses2 = []
                mi_losses2=[]
                action_losses2 = []
                dist_losses2 = []
                steps2 = []
                mean_reward2 = []
                mean_length2 = []
                mean_val_length2 = []
            elif line2.find('num timesteps') >= 0:
                timestep2 = int(re.match(r'.*?num timesteps ([0-9]*),.*', line2).group(1))
                if timestep2 % eps_step2 == 0 and timestep2 > 0:
                    eps2.append(timestep2 // eps_step2)
                    mean_reward2.append(np.nan)
                    mean_length2.append(np.nan)
                    mean_val_length2.append(np.nan)
                steps2.append(timestep2)
                critic_losses2.append(np.nan)
                actor_losses2.append(np.nan)
                value_losses2.append(np.nan)
                mi_losses2.append(np.nan)
                action_losses2.append(np.nan)
                dist_losses2.append(np.nan)
            if line2.find('Global Loss critic/actor:') >= 0:
                res2 = re.match(r'.*?Global Loss critic/actor: ([0-9-.]*)/([0-9-.]*).*', line2)
                critic_losses2[-1] = float(res2.group(1))
                actor_losses2[-1] = float(res2.group(2))
            if line2.find('Global Loss value/mi_loss/action/dist/predict:') >= 0:
                res2 = re.match(r'.*?Global Loss value/mi_loss/action/dist/predict: ([0-9-.]*)/([0-9-.]*)/([0-9-.]*)/([0-9-.]*)/.*', line2)
                value_losses2[-1] = float(res2.group(1))
                mi_losses2[-1]=float(res2.group(2))
                action_losses2[-1] = float(res2.group(3))
                dist_losses2[-1] = float(res2.group(2))+4.34164

    steps2 = np.asarray(steps2)
    g_steps2 = np.asarray(steps2)
    actor_losses2 = np.asarray(actor_losses2)
    critic_losses2 = np.asarray(critic_losses2)
    eps2 = np.asarray(eps2) / ppo_sample_eps2
    mean_reward2 = np.asarray(mean_reward2)
    mean_length2 = np.asarray(mean_length2)
    mean_val_length2 = np.asarray(mean_val_length2)
    value_losses2 = np.asarray(value_losses2)
    action_losses2 = np.asarray(action_losses2)
    dist_losses2 = np.asarray(dist_losses2)

    '''fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax[0].plot(steps, critic_losses, color='g')
    ax[1].plot(steps, actor_losses, color='r')
    ax[2].plot(eps, mean_reward, color='b')'''
    timestep = 0
    critic_losses = []
    actor_losses = []
    steps = []
    mean_reward = []
    eps_step = 0
    ppo_sample_eps = 0
    exp_name = 'exp'
    eps = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if len(line) < 5:
                continue
            line = line.rstrip()
            if 'Namespace(' in line:
                res = re.match(r'.*?exp_name=\'(.*?)\'.*?max_episode_length=([0-9]*).*?num_processes=([0-9]*).*?ppo_sample_eps=([0-9]*).*', line)
                exp_name = res.group(1)
                eps_step = int(res.group(2)) * int(res.group(3))
                ppo_sample_eps = int(res.group(4))
                eps = []
                critic_losses = []
                actor_losses = []
                value_losses = []
                mi_losses=[]
                action_losses = []
                dist_losses = []
                steps = []
                mean_reward = []
                mean_length = []
                mean_val_length = []
            elif line.find('num timesteps') >= 0:
                timestep = int(re.match(r'.*?num timesteps ([0-9]*),.*', line).group(1))
                if timestep % eps_step == 0 and timestep > 0:
                    eps.append(timestep // eps_step)
                    mean_reward.append(np.nan)
                    mean_length.append(np.nan)
                    mean_val_length.append(np.nan)
                steps.append(timestep)
                critic_losses.append(np.nan)
                actor_losses.append(np.nan)
                value_losses.append(np.nan)
                mi_losses.append(np.nan)
                action_losses.append(np.nan)
                dist_losses.append(np.nan)
            if line.find('Global Loss critic/actor:') >= 0:
                res = re.match(r'.*?Global Loss critic/actor: ([0-9-.]*)/([0-9-.]*).*', line)
                critic_losses[-1] = float(res.group(1))
                actor_losses[-1] = float(res.group(2))
            if line.find('Global Loss value/mi_loss/action/dist/predict:') >= 0:
                res = re.match(r'.*?Global Loss value/mi_loss/action/dist/predict: ([0-9-.]*)/([0-9-.]*)/([0-9-.]*)/([0-9-.]*)/.*', line)
                value_losses[-1] = float(res.group(1))
                mi_losses[-1]=float(res.group(2))
                action_losses[-1] = float(res.group(3))
                dist_losses[-1] = float(res.group(2))+4.34164
            if line.find('Global eps mean/med/min/max eps rew:') >= 0:
                if mean_reward and mean_reward[-1] is np.nan:
                    mean_reward[-1] = float(re.match(r'.*?Global eps mean/med/min/max eps rew: ([0-9-.]*).*', line).group(1))
            if line.find('Global eps mean/med eps len:') >= 0:
                if mean_length and mean_length[-1] is np.nan:
                    mean_length[-1] = int(re.match(r'.*?Global eps mean/med eps len: ([0-9-.]*).*', line).group(1))
            if line.find('Validation eps mean/med eps len:') >= 0:
                if mean_val_length and mean_val_length[-1] is np.nan:
                    mean_val_length[-1] = int(re.match(r'.*?Validation eps mean/med eps len: ([0-9-.]*).*', line).group(1))

    steps = np.asarray(steps)
    g_steps = np.asarray(steps)
    actor_losses = np.asarray(actor_losses)
    critic_losses = np.asarray(critic_losses)
    eps = np.asarray(eps) / ppo_sample_eps
    mean_reward = np.asarray(mean_reward)
    mean_length = np.asarray(mean_length)
    mean_val_length = np.asarray(mean_val_length)
    value_losses = np.asarray(value_losses)
    action_losses = np.asarray(action_losses)
    dist_losses = np.asarray(dist_losses)

    fig, ax = plt.subplots(1, 1)
    ax.plot(g_steps/20000, dist_losses, color='r',label='Complete AIM-Mapping MI')
    ax.plot(g_steps2/20000, dist_losses2, color='g',label='Ablated Privileged Representation MI')
    ax.set_xlabel('Training rounds')
    plt.legend()
    plt.savefig(exp_name + '.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/NeuralCoMapping/train/models/infonce_920_mi_1_0/basic.log')
    parser.add_argument('--is_dir', action='store_true', default=False)
    args = parser.parse_args()
    if args.is_dir:
        for file in os.listdir(args.path):
            if file[-4:] == '.log':
                main(os.path.join(args.path, file))
    else:
        main(args.path,'/home/NeuralCoMapping/train/models/infonce_923_nodifference/basic.log')