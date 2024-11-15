import re 
import matplotlib.pyplot as plt
log_file="/home/NeuralCoMapping/train/models/infonce_920_mi_1_0/basic.log"
with open(log_file,'r')as file:
    log_data=file.readlines()
timesteps_pattern= r"num timesteps (\d+)"
mean_eps= r"Global eps mean/med eps len: (\d+)/\d+"
timestep=[]
mean_lens=[]
for line in log_data:

    timestep_mateh=re.search(timesteps_pattern,line)
    mean_mateh=re.search(mean_eps,line)
    if timestep_mateh :
        timestep.append(int(timestep_mateh.group(1))/20000)
    if mean_mateh:
        mean_lens.append(int(mean_mateh.group(1)))

with open("/home/NeuralCoMapping/train/models/Asymmetric_NLMinus_AC_Cluster_LowLr/basic.log",'r')as file:
    log_data=file.readlines()
timesteps_pattern= r"num timesteps (\d+)"
mean_eps= r"Global eps mean/med eps len: (\d+)/\d+"
timestep2=[]
mean_lens2=[]
for line in log_data:

    timestep_mateh=re.search(timesteps_pattern,line)
    mean_mateh=re.search(mean_eps,line)
    if timestep_mateh :
        timestep2.append(int(timestep_mateh.group(1))/20000)
    if mean_mateh:
        mean_lens2.append(int(mean_mateh.group(1)))

with open("/home/NeuralCoMapping/train/models/infonce_923_nodifference/basic.log",'r')as file:
    log_data=file.readlines()
timesteps_pattern= r"num timesteps (\d+)"
mean_eps= r"Global eps mean/med eps len: (\d+)/\d+"
timestep3=[]
mean_lens3=[]
for line in log_data:

    timestep_mateh=re.search(timesteps_pattern,line)
    mean_mateh=re.search(mean_eps,line)
    if timestep_mateh :
        timestep3.append(int(timestep_mateh.group(1))/20000)
    if mean_mateh:
        mean_lens3.append(int(mean_mateh.group(1)))



plt.plot(timestep,mean_lens,label='AIM-Mapping')
plt.plot(timestep2,mean_lens2,label='Ablated privileged representation')
plt.plot(timestep3,mean_lens3,label='Ablated mutual information evaluation')
plt.xlabel('Training rounds')
plt.ylabel('Time step')
plt.legend()

plt.show()