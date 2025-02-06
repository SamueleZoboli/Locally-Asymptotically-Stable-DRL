# README

## Locally Asymptotically Stable Deep Actor-Critic Algorithms for Quadratic Cost Setpoint Optimal Control Problems

This repository provides the necessary steps to run and evaluate the experiments presented in Section 1.2 of ["Achieving reliable control : robustness and stability in
nonlinear systems via DNN-based feedback design"](https://theses.hal.science/tel-04631894/file/TH2023ZOBOLISAMUELE.pdf). An extract is proposed in summary.pdf. Some files are modified versions of [gym](https://github.com/openai/gym) and [PyBullet](https://pybullet.org/wordpress/) files.

### Prerequisites

The following steps assume an Ubuntu environment and were tested on Ubuntu 20.04.

## Installation

### 1. Create a Virtual Environment (Python 3.8)

#### Using Conda:
```sh
conda create --name <env_name> python=3.8
conda activate <env_name>
```

#### Using venv on Ubuntu:
```sh
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
python3.8 -m venv <name-of-the-env>
source <name-of-the-env>/bin/activate
```

### 2. Install PyTorch
Follow the instructions on the [website](https://pytorch.org/).

### 3. Install Required Packages
```sh
pip install -r <path-to>/requirements.txt
```

### 4. Replace Required Folders
Replace the following folders in your virtual environment:
```
<path>/<to>/<virtual-env>/lib/python3.8/site-packages/
```
Folders to replace:
- `gym/envs/classic_control`
- `gym/envs/__init__`
- `pybullet_data/mjcf`
- `pybullet_envs`
- `stable_baselines3`

## Running Experiments

### 5. Run Training
Use `train.py` with the following required arguments:

```sh
python train.py --algo <ppo|ddpg|sac|td3> --env <pendulum|cartpole|cartpole_double> --learning_steps <num_steps>
```

#### Optional Arguments:
- `--tensorboard_logs` : Enable TensorBoard logging at `logs/<env>/<algo>/tensorboard`
- `--las` : Use the LAS version of the algorithm
- `--learning_rate xey` : Set learning rate
- `--seed x` : Set the random seed
- `--policy_kwargs` : Set network parameters (e.g., `net_arch=[dict(pi=[..],vf=[..])], activation_fn=nn.Tanh`)

Example:
```sh
python train.py --algo ddpg --env cartpole --learning_steps 2e6
python train.py --algo ppo --env pendulum --learning_steps 1e6 --tensorboard_logs --learning_rate 1e-3 --seed 0 --las --policy_kwargs net_arch=[dict(pi=[256,256],vf=[256,256])], activation_fn=nn.Tanh
```

### 6. Visualizing Results
#### Using TensorBoard:
```sh
tensorboard --logdir <path-to>/logs/<env>/<algo>/tensorboard/
```
#### Checking Logs Manually:
- Stability results (far initial condition): `<path-to>/logs/<env>/<algo>/stability/<seed>/far`
- Stability results (close initial condition): `<path-to>/logs/<env>/<algo>/stability/<seed>/eq`
- Performance results: `<path-to>/logs/<env>/<algo>/performances/<seed>/`

### 7. Evaluating a Model
Use `eval.py` with the following required arguments:

```sh
python eval.py --algo <ppo|ddpg|sac|td3> --env <pendulum|cartpole|cartpole_double> --model <path-to>/trained_models/<algo>/<env>.zip
```

#### Optional Argument:
- `--n_eval_ep x` : Number of evaluation episodes

Example:
```sh
python eval.py --algo ddpg --env cartpole_double --model <path-to>/trained_models/ddpg/cartpole_double.zip
python eval.py --algo ddpg --env cartpole_double --model <path-to>/trained_models/las_ddpg/cartpole_double.zip --n_eval_ep 15
```

### 8. Running Experiments from the Paper
#### PSU:
```sh
python train.py --algo ppo --env pendulum --learning_steps 2.5e6 --tensorboard_logs --seed 0 --learning_rate 3e-3 --las --policy_kwargs net_arch=[dict(pi=[64,64],vf=[64,64])], activation_fn=nn.Tanh
```

#### IPSU:
```sh
python train.py --algo ppo --env cartpole --learning_steps 2.5e6 --tensorboard_logs --seed 0 --learning_rate 2.5e-4 --las --policy_kwargs net_arch=[dict(pi=[256,256],vf=[256,256])], activation_fn=nn.Tanh
```

#### DIPSU:
```sh
python train.py --algo ppo --env cartpole_double --learning_steps 1e7 --tensorboard_logs --seed 0 --learning_rate 1e-4 --las --policy_kwargs net_arch=[dict(pi=[64,64],vf=[64,64])], activation_fn=nn.Tanh
```



