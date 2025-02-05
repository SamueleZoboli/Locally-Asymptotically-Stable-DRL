import argparse
import gym
import json
import torch as th
import torch.nn as nn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3 import A2C, PPO, SAC, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import ActionNoise
from typing import Any, Callable, Dict, Optional, Union

NOISY_EVAL = True


def create_agent(algorithm: str,
                 env_name: str,
                 las: bool = False,
                 n_envs: int = 1,
                 learning_rate: Union[float, Callable] = 1e-3,
                 batch_size: int = 100,
                 gamma: float = 0.99,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Dict[str, Any] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,

                 # on-policy params
                 n_steps: int = 2048,
                 n_epochs: int = 10,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 rms_prop_eps: float = 1e-5,
                 use_rms_prop: bool = True,
                 normalize_advantage: bool = False,

                 # off-policy params
                 buffer_size: int = int(1e6),
                 learning_starts: int = 100,
                 tau: float = 0.005,
                 train_freq: int = 1,
                 gradient_steps: int = -1,
                 action_noise: Optional[ActionNoise] = None,
                 optimize_memory_usage: bool = False,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 target_update_interval: int = 1,
                 target_entropy: Union[str, float] = "auto",
                 use_sde_at_warmup: bool = False, ):
    """Defines an algorithm between 'a2c','ppo','sac','ddpg','td3' and the environment. Then selects the standard
     formulation or the Locally Asymptothically Stable one and builds the correct environment and agent. The parameters
     are the ones of Stable Baselines 3"""

    envs = {"pendulum": 'Pendulum-v0',
            'cartpole': 'InvertedPendulumSwingupBulletEnv-v0',
            'cartpole_double': 'InvertedDoublePendulumBulletEnv-v0',
            }
    assert env_name in envs, "Not a supported environment"

    if n_envs == 1:
        env = Monitor(gym.make(envs[env_name]))
    else:
        env = make_vec_env(envs[env_name], n_envs)

    if las:
        policy = 'AmlpPolicy'
        if n_envs == 1:
            env.set_origin_stop(True)
        else:
            for e in env.envs:
                e.set_origin_stop(True)
    else:
        policy = 'MlpPolicy'

    assert algorithm in ['a2c', 'ppo', 'sac', 'ddpg', 'td3'], "Not a supported algorithm"

    if algorithm == 'ddpg':
        model = DDPG(policy=policy, env=env, learning_rate=learning_rate, buffer_size=buffer_size,
                     learning_starts=learning_starts, batch_size=batch_size, tau=tau,
                     gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps,
                     action_noise=action_noise,
                     optimize_memory_usage=optimize_memory_usage,
                     tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                     policy_kwargs=policy_kwargs,
                     verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)
    elif algorithm == 'td3':
        model = TD3(policy=policy, env=env, learning_rate=learning_rate, buffer_size=buffer_size,
                    learning_starts=learning_starts, batch_size=batch_size, tau=tau,
                    gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps,
                    action_noise=action_noise,
                    optimize_memory_usage=optimize_memory_usage, policy_delay=policy_delay,
                    target_policy_noise=target_policy_noise, target_noise_clip=target_noise_clip,
                    tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                    policy_kwargs=policy_kwargs,
                    verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)
    elif algorithm == 'a2c':
        model = A2C(policy=policy, env=env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma,
                    gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    rms_prop_eps=rms_prop_eps, use_rms_prop=use_rms_prop, use_sde=use_sde,
                    sde_sample_freq=sde_sample_freq, normalize_advantage=normalize_advantage,
                    tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                    policy_kwargs=policy_kwargs,
                    verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)
    elif algorithm == 'ppo':
        model = PPO(policy=policy, env=env, learning_rate=learning_rate, n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                    clip_range_vf=clip_range_vf, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    use_sde=use_sde, sde_sample_freq=sde_sample_freq, target_kl=target_kl,
                    tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                    policy_kwargs=policy_kwargs,
                    verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)
    elif algorithm == 'sac':
        model = SAC(policy=policy, env=env, learning_rate=learning_rate, buffer_size=buffer_size,
                    learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                    train_freq=train_freq, gradient_steps=gradient_steps,
                    action_noise=action_noise, optimize_memory_usage=optimize_memory_usage, ent_coef=ent_coef,
                    target_update_interval=target_update_interval, target_entropy=target_entropy,
                    use_sde=use_sde, sde_sample_freq=sde_sample_freq, use_sde_at_warmup=use_sde_at_warmup,
                    tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                    policy_kwargs=policy_kwargs,
                    verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)

    return model, env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train', description='Define and train a DeepRL agent', allow_abbrev=False)

    parser.add_argument('--algo', type=str, help='algorithm to be used', required=True)
    parser.add_argument('--env', type=str, help='environment to be used', required=True)
    parser.add_argument('--learning_steps', metavar='steps', type=float, help='number of training steps', required=True)

    parser.add_argument('--las', action='store_true', help='use local controller')
    parser.add_argument('--n_envs', type=int, help=' number of training environments', default=1)
    parser.add_argument('--learning_rate', type=float, help=' learning rate', default=1e-3)
    parser.add_argument('--batch_size', type=int, help=' size of the batch ', default=128)
    parser.add_argument('--gamma', type=float, help=' discount factor ', default=0.99)
    parser.add_argument('--tensorboard_log', action='store_true', help=' create logs using tensorboard ')
    parser.add_argument('--create_eval_env', action='store_true', help=' use an evaluation environment ')
    parser.add_argument('--policy_kwargs', type=str,
                        help=' net architecture, activation functions etc..., separate params with ++, e.g. net_arch=[dict\(pi=[32,32]\)]++activation=\"nn.Tanh\" ',
                        default=None)  # the net_arch requires net_arch=[dict(pi=...,vf=...)] for onpolicy algorithms while net_arch=dict(pi=...,qf=...) for offpolicy ones if you want to avoid net sharing for onpolicy algo
    parser.add_argument('--verbose', type=int, help=' verbosity level ', default=0)
    parser.add_argument('--seed', type=int, help=' random seed ', default=0)
    parser.add_argument('--device', type=str, help=' cpu or cuda ', default='auto')
    parser.add_argument('--_init_setup_model', action='store_false')
    parser.add_argument('--n_steps', type=int, help=' on policy algorithm parameter', default=2048)
    parser.add_argument('--n_epochs', type=int, help=' on policy algorithm parameter', default=10)
    parser.add_argument('--gae_lambda', type=float, help=' on policy algorithm parameter', default=0.95)
    parser.add_argument('--clip_range', type=float, help=' on policy algorithm parameter', default=0.2)
    parser.add_argument('--clip_range_vf', type=float, help=' on policy algorithm parameter', default=None)
    parser.add_argument('--ent_coef', type=float, help=' on policy algorithm parameter', default=0.0)
    parser.add_argument('--vf_coef', type=float, help=' on policy algorithm parameter', default=0.5)
    parser.add_argument('--max_grad_norm', type=float, help=' on policy algorithm parameter', default=0.5)
    parser.add_argument('--use_sde', action='store_true', help=' on policy algorithm parameter')
    parser.add_argument('--sde_sample_freq', type=int, help=' on policy algorithm parameter', default=-1)
    parser.add_argument('--target_kl', type=float, help=' on policy algorithm parameter', default=None)
    parser.add_argument('--rms_prop_eps', type=float, help=' on policy algorithm parameter', default=1e-5)
    parser.add_argument('--use_rms_prop', action='store_false', help=' on policy algorithm parameter')
    parser.add_argument('--normalize_advantage', action='store_true', help=' on policy algorithm parameter')
    parser.add_argument('--buffer_size', type=int, help=' off policy algorithm parameter', default=int(1e6))
    parser.add_argument('--learning_starts', type=int, help=' off policy algorithm parameter', default=100)
    parser.add_argument('--tau', type=float, help=' off policy algorithm parameter', default=0.005)
    parser.add_argument('--train_freq', type=int, help=' off policy algorithm parameter', default=1)
    parser.add_argument('--gradient_steps', type=int, help=' off policy algorithm parameter', default=-1)
    parser.add_argument('--action_noise', type=str, help=' off policy algorithm parameter', default=None)
    parser.add_argument('--optimize_memory_usage', action='store_true', help=' off policy algorithm parameter')
    parser.add_argument('--policy_delay', type=int, help=' off policy algorithm parameter', default=2)
    parser.add_argument('--target_policy_noise', type=float, help=' off policy algorithm parameter', default=0.2)
    parser.add_argument('--target_noise_clip', type=float, help=' off policy algorithm parameter', default=0.5)
    parser.add_argument('--target_update_interval', type=int, help=' off policy algorithm parameter', default=1)
    parser.add_argument('--target_entropy', type=str, help=' off policy algorithm parameter', default='auto')
    parser.add_argument('--use_sde_at_warmup', action='store_true', help=' off policy algorithm parameter')
    parser.add_argument('--model_save_freq', type=float, help=' steps between model saves', default=10000)
    parser.add_argument('--noisy_eval_freq', type=float, help=' steps between model noisy env evaluation', default=10000)
    parser.add_argument('--perf_eval_freq', type=float, help=' steps between model evaluation in perfect env', default=10000)
    parser.add_argument('--n_eval_ep', type=int, help=' number of evaluation episodes (both noisy and perfect) ', default=10)


    args = parser.parse_args()

    algorithm = args.algo
    env_name = args.env
    learning_steps = args.learning_steps
    las = args.las
    n_envs = args.n_envs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    gamma = args.gamma
    tensorboard_log = args.tensorboard_log
    create_eval_env = args.create_eval_env
    if args.policy_kwargs is not None:
        kwargs = '{\"' + args.policy_kwargs + '}'
        start_idx = kwargs.find('activation_fn=') + len('activation_fn=')
        if start_idx != len('activation_fn=') - 1:
            end_idx = kwargs[start_idx:].find('++')
            kwargs = kwargs[:start_idx] + '\"' + kwargs[start_idx:end_idx] + '\"' + kwargs[end_idx:]
        kwargs = kwargs.replace('=', '\":')
        kwargs = kwargs.replace('++', ',\"')
        if algorithm == 'ppo' or algorithm == 'a2c':
            start_idx = kwargs.find('net_arch":[dict(') + len('net_arch":[dict(')
            if start_idx != len('net_arch":[dict(') - 1:
                end_idx = kwargs[start_idx:].find(')') + start_idx
                slice1 = kwargs[:start_idx]
                slice2 = kwargs[start_idx:end_idx]
                slice3 = kwargs[end_idx:]
                slice2 = slice2.replace('\":', '=')
                kwargs = slice1 + slice2 + slice3
                kwargs.find('net_arch":[dict(') + len('net_arch":[dict(')
                start_idx = kwargs.find('net_arch":[') + len('net_arch":[')
                end_idx = len('dict(') + len(slice2) + len(')]}') + start_idx
                strdict = kwargs[start_idx:end_idx - len(']}')]
                kwargs = kwargs[:start_idx] + json.dumps(eval(strdict)) + kwargs[end_idx - len(']}'):]
        elif algorithm == 'sac' or algorithm == 'td3' or algorithm == 'ddpg':
            start_idx = kwargs.find('net_arch":dict(') + len('net_arch":dict(')
            if start_idx != len('net_arch":dict(') - 1:
                end_idx = kwargs[start_idx:].find(')') + start_idx
                slice1 = kwargs[:start_idx]
                slice2 = kwargs[start_idx:end_idx]
                slice3 = kwargs[end_idx:]
                slice2 = slice2.replace('\":', '=')
                kwargs = slice1 + slice2 + slice3
                kwargs.find('net_arch":dict(') + len('net_arch":dict(')
                start_idx = kwargs.find('net_arch":') + len('net_arch":')
                end_idx = len('dict(') + len(slice2) + len(')}') + start_idx
                strdict = kwargs[start_idx:end_idx - len('}')]
                kwargs = kwargs[:start_idx] + json.dumps(eval(strdict)) + kwargs[end_idx - len('}'):]
        policy_kwargs = json.loads(kwargs)
        if 'activation_fn' in policy_kwargs:
            policy_kwargs['activation_fn'] = eval(policy_kwargs['activation_fn'])
    else:
        policy_kwargs = None
    verbose = args.verbose
    seed = args.seed
    device = args.device
    _init_setup_model = args._init_setup_model
    n_steps = args.n_steps
    n_epochs = args.n_epochs
    gae_lambda = args.gae_lambda
    clip_range = args.clip_range
    clip_range_vf = args.clip_range_vf
    ent_coef = args.ent_coef
    vf_coef = args.vf_coef
    max_grad_norm = args.max_grad_norm
    use_sde = args.use_sde
    sde_sample_freq = args.sde_sample_freq
    target_kl = args.target_kl
    rms_prop_eps = args.rms_prop_eps
    use_rms_prop = args.use_rms_prop
    normalize_advantage = args.normalize_advantage
    buffer_size = args.buffer_size
    learning_starts = args.learning_starts
    tau = args.tau
    train_freq = args.train_freq
    gradient_steps = args.gradient_steps
    action_noise = args.action_noise
    optimize_memory_usage = args.optimize_memory_usage
    policy_delay = args.policy_delay
    target_policy_noise = args.target_policy_noise
    target_noise_clip = args.target_noise_clip
    target_update_interval = args.target_update_interval
    target_entropy = args.target_entropy
    use_sde_at_warmup = args.use_sde_at_warmup
    model_save_freq = int(args.model_save_freq)
    noisy_eval_freq = int(args.noisy_eval_freq)
    perfect_eval_freq = int(args.perf_eval_freq)
    n_eval_episodes = args.n_eval_ep


    d = {'algorithm': algorithm, 'env_name': env_name, 'learning_steps': learning_steps, 'las': las, 'n_envs': n_envs,
         'learning_rate': learning_rate, 'batch_size': batch_size, 'gamma': gamma, 'tensorboard_log': tensorboard_log,
         'create_eval_env': create_eval_env, 'policy_kwargs': policy_kwargs, 'verbose': verbose, 'seed': seed,
         'device': device, '_init_setup_model': _init_setup_model, 'n_steps': n_steps, 'n_epochs': n_epochs,
         'gae_lambda': gae_lambda, 'clip_range': clip_range, 'clip_range_vf': clip_range_vf, 'ent_coef': ent_coef,
         'vf_coef': vf_coef, 'max_grad_norm': max_grad_norm, 'use_sde': use_sde,
         'sde_sample_freq': sde_sample_freq, 'target_kl': target_kl, 'rms_prop_eps': rms_prop_eps,
         'use_rms_prop': use_rms_prop, 'normalize_advantage': normalize_advantage,
         'buffer_size': buffer_size, 'learning_starts': learning_starts, 'tau': tau, 'train_freq': train_freq,
         'gradient_steps': gradient_steps, 'action_noise': action_noise,
         'optimize_memory_usage': optimize_memory_usage, 'policy_delay': policy_delay,
         'target_policy_noise': target_policy_noise, 'target_noise_clip': target_noise_clip,
         'target_update_interval': target_update_interval, 'target_entropy': target_entropy,
         'use_sde_at_warmup': use_sde_at_warmup, 'model_save_freq': model_save_freq, 'noisy_eval_freq': noisy_eval_freq}

    las_name = 'las_' if las else ''

    if tensorboard_log:
        log_dir = './logs/' + env_name + '/' + las_name + algorithm + '/tensorboard/seed_' + str(seed)
    else:
        log_dir = None

    agent, env = create_agent(algorithm=algorithm, env_name=env_name, las=las, n_envs=n_envs,
                              learning_rate=learning_rate,
                              batch_size=batch_size, gamma=gamma, tensorboard_log=log_dir,
                              create_eval_env=create_eval_env, policy_kwargs=policy_kwargs, verbose=verbose, seed=seed,
                              device=device, _init_setup_model=_init_setup_model, n_steps=n_steps, n_epochs=n_epochs,
                              gae_lambda=gae_lambda, clip_range=clip_range, clip_range_vf=clip_range_vf,
                              ent_coef=ent_coef,
                              vf_coef=vf_coef, max_grad_norm=max_grad_norm, use_sde=use_sde,
                              sde_sample_freq=sde_sample_freq, target_kl=target_kl, rms_prop_eps=rms_prop_eps,
                              use_rms_prop=use_rms_prop, normalize_advantage=normalize_advantage,
                              buffer_size=buffer_size,
                              learning_starts=learning_starts, tau=tau, train_freq=train_freq,
                              gradient_steps=gradient_steps,
                              action_noise=action_noise, optimize_memory_usage=optimize_memory_usage,
                              policy_delay=policy_delay, target_policy_noise=target_policy_noise,
                              target_noise_clip=target_noise_clip, target_update_interval=target_update_interval,
                              target_entropy=target_entropy, use_sde_at_warmup=use_sde_at_warmup)

    if n_envs > 1:
        noisy_env_name = env.envs[0].spec.id.partition('-')[0] + 'Noisy-v0'
        perf_env_name = env.envs[0].spec.id
    else:
        noisy_env_name = env.env.spec.id.partition('-')[0] + 'Noisy-v0'
        perf_env_name = env.env.spec.id

    checkpoint_callback = CheckpointCallback(save_freq=int(model_save_freq),
                                             save_path='./logs/' + env_name + '/' + las_name + algorithm + '/models/seed_' + str(
                                                 seed),
                                             name_prefix=las_name + algorithm)
    
    eval_env  = Monitor(gym.make(perf_env_name))
    eval_callback = EvalCallback(eval_env,
                                log_path='./logs/' + env_name + '/' + las_name + algorithm + '/performances/seed_' + str(
                                    seed),
                                n_eval_episodes=n_eval_episodes,
                                eval_freq=perfect_eval_freq,
                                best_model_save_path='./logs/' + env_name + '/' + las_name + algorithm + '/performances/seed_' + str(
                                    seed) + '/best',
                                deterministic=True, render=False, verbose=0)

    if NOISY_EVAL:
        eval_env_1 = Monitor(gym.make(noisy_env_name))
        eval_env_1.set_start_from_eq(False)
        eval_callback_1 = EvalCallback(eval_env_1,
                                       log_path='./logs/' + env_name + '/' + las_name + algorithm + '/stability/seed_' + str(
                                           seed) + '/far',
                                       eval_freq=noisy_eval_freq,
                                       n_eval_episodes=n_eval_episodes,
                                       deterministic=True, render=False, verbose=0)
        eval_env_2 = Monitor(gym.make(noisy_env_name))
        eval_env_2.set_start_from_eq(True)
        eval_callback_2 = EvalCallback(eval_env_2,
                                       log_path='./logs/' + env_name + '/' + las_name + algorithm + '/stability/seed_' + str(
                                           seed) + '/eq',
                                       eval_freq=noisy_eval_freq,
                                       n_eval_episodes=n_eval_episodes,
                                       deterministic=True, render=False, verbose=0)
        callback = CallbackList([checkpoint_callback, eval_callback, eval_callback_1, eval_callback_2])
    else:
        callback = CallbackList([checkpoint_callback, eval_callback])

    if args.policy_kwargs is None:
        pol_kw = ''
    else:
        pol_kw = '_' + args.policy_kwargs

    agent.learn(learning_steps, callback=callback,
                tb_log_name=las_name + algorithm + '_seed_' + str(seed) + '_lr_' + str(
                    learning_rate) + pol_kw)

    env.close()
    eval_env.close()

    if NOISY_EVAL:
        eval_env_1.close()
        eval_env_2.close()
