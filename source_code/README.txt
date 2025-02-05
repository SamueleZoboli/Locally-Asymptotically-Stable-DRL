We provide the steps necessary to run and evaluate the experiments presented in the paper "Locally Asymptotically Stable Deep Actor-Critic Algorithms For Quadratic Cost Setpoint Optimal Control Problems". All those steps assume Ubuntu and were tested on Ubuntu 20.04

1) create a virtual environment with python 3.8 and activate it

with conda:   conda create --name <env_name> python=3.8
			  conda activate <env_name>
			  
with venv on Ubuntu:    sudo add-apt-repository ppa:deadsnakes/ppa
						sudo apt-get update
						sudo apt-get install python3.8
						python3.8 -m venv <name-of-the-env>
			  			venv <name-of-the-env>/bin/activate

2) install pytorch: https://pytorch.org/

3) install requirements :
pip install -r <path-to>/requirements.txt


4) replace the folders gym/envs/classic_control, gym/envs/__init__ , pybullet_data/mjcf, pybullet_envs, stable_baselines3 with the ones provided.
	These folders to be replaced should be in <path>/<to>/<virtual-env-of-choice>/lib/python3.8/site-packages/

5) to run experiments use train.py 

train.py requires three arguments:
--algo (the algorithm to be used: ppo,ddpg,sac,td3)
--env (the environment: pendulum, cartpole, cartpole_double)
--learning_steps (number of steps used for training: xe^y, e.g., 1e6)

train.py has some optional arguments, the most relevant are:
--tensorboard_logs  : flag to generate logs for tensorboard in the folder logs/<env>/<algo>/tensorboard
--las               : flag to use the LAS version of the algorithm
--learning_rate xey: learning rate to use (xey)
--seed x			: sets the random seed (x)
--policy_kwargs  net_arch=...++activation_fn=nn.<act_fn>   : set network parameters; ppo requires net_arch=[dict\(pi=[..],vf=[..]\)], sac/ddpg/td3 require net_arch=dict\(pi=[..],qf=[..]\)

e.g. python train.py --algo ddpg --env cartpole --learning_steps 2e6 
e.g. python train.py --algo ppo --env pendulum --learning_steps 1e6 --tensorboard_log --learning_rate 1e-3 --seed 0 --las --policy_kwargs net_arch=[dict\(pi=[256,256],vf=[256,256]\)]++activation_fn=nn.Tanh


6) the results can be visualized with tensorboard by using:
tensorboard --logdir <path-to>/logs/<env>/<algo>/tensorboard/

 alternatively, results can can be obtained by browsing the automatically generated log folder (data will be stored as "evaluations.npz"):
  to obtain the stability results for an initial condition outside of the domain of attraction of the local component navigate to: <path-to>/logs/<env>/<algo>/stability/<seed>/far
  to obtain the stability results for an initial condition close to the setpoint navigate to: <path-to>/logs/<env>/<algo>/stability/<seed>/eq
  to obtain the performances results navigate to: <path-to>/logs/<env>/<algo>/performances/<seed>/
  
7) to evaluate a model use eval.py 

eval.py requires three arguments:
--algo (the algorithm to be used: ppo,ddpg,sac,td3)
--env (the environment: pendulum, cartpole, cartpole_double)
--model (the path to the model.zip file)

eval.py has an optional argument:
--n_eval_ep  x : number of evaluation episodes (x)

e.g. python eval.py --algo ddpg --env cartpole_double --model <path-to>/trained_models/ddpg/cartpole_double.zip 
e.g. python eval.py --algo ddpg --env cartpole_double --model <path-to>/trained_models/las_ddpg/cartpole_double.zip --n_eval_ep 15

Some pretrained models can be found at https://we.tl/t-g05hwPkzZE

5b) to run the experiments in the paper (only one random seed):
PSU:

	python train.py --algo ppo --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 3e-3 --las --policy_kwargs net_arch=[dict\(pi=[64,64],vf=[64,64]\)]++activation_fn=nn.Tanh
	python train.py --algo ppo --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 3e-3 --policy_kwargs net_arch=[dict\(pi=[64,64],vf=[64,64]\)]++activation_fn=nn.Tanh
	python train.py --algo ddpg --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --las --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo ddpg --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo td3 --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --las --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo td3 --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo sac --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --las --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.ReLU
	python train.py --algo sac --env pendulum --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.ReLU
	
IPSU:

	python train.py --algo ppo --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 2.5e-4 --las --policy_kwargs net_arch=[dict\(pi=[256,256],vf=[256,256]\)]++activation_fn=nn.Tanh
	python train.py --algo ppo --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 2.5e-4 --policy_kwargs net_arch=[dict\(pi=[256,256],vf=[256,256]\)]++activation_fn=nn.Tanh
	python train.py --algo ddpg --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --las --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo ddpg --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo td3 --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --las --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo td3 --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo sac --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --las --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.ReLU
	python train.py --algo sac --env cartpole --learning_steps 2.5e6 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.ReLU
	
DIPSU: 
	python train.py --algo ppo --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 1e-4 --las --policy_kwargs net_arch=[dict\(pi=[64,64],vf=[64,64]\)]++activation_fn=nn.Tanh
	python train.py --algo ppo --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 1e-4 --policy_kwargs net_arch=[dict\(pi=[64,64],vf=[64,64]\)]++activation_fn=nn.Tanh
	python train.py --algo ddpg --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 1e-4 --las --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.Tanh
	python train.py --algo ddpg --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 1e-4 --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo td3 --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 2e-4 --las --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.Tanh
	python train.py --algo td3 --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 6e-4 --policy_kwargs net_arch=dict\(pi=[400,300],qf=[400,300]\)++activation_fn=nn.ReLU
	python train.py --algo sac --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 1e-4 --las --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.Tanh
	train.py --algo sac --env cartpole_double --learning_steps 1e7 --tensorboard_log --seed 0 --learning_rate 1e-3 --policy_kwargs net_arch=dict\(pi=[256,256],qf=[256,256]\)++activation_fn=nn.ReLU
