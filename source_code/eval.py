import argparse
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO, SAC, DDPG, TD3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval', description='Evaluate a model in perfect and corrupted environments',
                                     allow_abbrev=False)

    parser.add_argument('--algo', type=str, help='algorithm to be used', required=True)
    parser.add_argument('--env', type=str, help='environment to be used', required=True)
    parser.add_argument('--model', type=str, help='model path', required=True)

    parser.add_argument('--n_eval_ep', type=int, help=' number of evaluation episodes', default=10)

    args = parser.parse_args()

    algorithm = args.algo
    env_name = args.env
    model_path = args.model
    n_eval_episodes = args.n_eval_ep

    envs = {"pendulum": 'Pendulum-v0',
            'cartpole': 'InvertedPendulumSwingupBulletEnv-v0',
            'cartpole_double': 'InvertedDoublePendulumBulletEnv-v0',
            }
    assert env_name in envs, "Not a supported environment"

    assert algorithm in ['a2c', 'ppo', 'sac', 'ddpg', 'td3'], "Not a supported algorithm"

    # generate environments
    perf_env = Monitor(gym.make(envs[env_name]))
    perf_env.set_origin_stop(False)

    noisy_env_name = perf_env.env.spec.id.partition('-')[0] + 'Noisy-v0'

    stab_env_1 = Monitor(gym.make(noisy_env_name))
    stab_env_1.set_start_from_eq(True)
    stab_env_2 = Monitor(gym.make(noisy_env_name))
    stab_env_2.set_start_from_eq(False)

    # load agent

    if algorithm == 'ppo':
        agent = PPO.load(path=model_path, env=perf_env)
    elif algorithm == 'sac':
        agent = SAC.load(path=model_path, env=perf_env)
    elif algorithm == 'td3':
        agent = TD3.load(path=model_path, env=perf_env)
    elif algorithm == 'ddpg':
        agent = DDPG.load(path=model_path, env=perf_env)
    else:
        agent = A2C.load(path=model_path, env=perf_env)
        
    if isinstance(agent.policy ,agent.augmented_policies):
        algorithm = "las_" + algorithm

    # evaluate the agent

    rew_vec = np.zeros((n_eval_episodes,))
    stab1_vec = np.zeros((n_eval_episodes,))
    stab2_vec = np.zeros((n_eval_episodes,))

    np.random.seed() #avoid fixing the seed with load functions
    seed_vec = np.random.randint(10000, size=n_eval_episodes)

    for k in range(n_eval_episodes):
        seed = seed_vec[k].item()
        perf_env.seed(seed)
        stab_env_1.seed(seed)
        stab_env_2.seed(seed)

        perf_max_ep_len = perf_env.spec.max_episode_steps
        stab1_max_ep_len = stab_env_1.spec.max_episode_steps
        stab2_max_ep_len = stab_env_2.spec.max_episode_steps

        rew_sum = 0
        stab1_sum = 0
        stab2_sum = 0

        # collect trajectories
        obs = perf_env.reset()
        for i in range(perf_max_ep_len):
            u = agent.predict(obs, deterministic=True)
            obs, rew, done, _ = perf_env.step(u[0])
            rew_sum = rew_sum + rew

        obs = stab_env_1.reset()
        for i in range(stab1_max_ep_len):
            u = agent.predict(obs, deterministic=True)
            obs, rew, done, _ = stab_env_1.step(u[0])
            stab1_sum = stab1_sum + rew

        obs = stab_env_2.reset()
        for i in range(stab2_max_ep_len):
            u = agent.predict(obs, deterministic=True)
            obs, rew, done, _ = stab_env_2.step(u[0])
            stab2_sum = stab2_sum + rew

        rew_vec[k] = rew_sum
        stab1_vec[k] = stab1_sum
        stab2_vec[k] = stab2_sum

    # compute mean and std

    perf_mean, perf_std = np.mean(rew_vec), np.std(rew_vec)
    stab1_mean, stab1_std = np.mean(stab1_vec), np.std(stab1_vec)
    stab2_mean, stab2_std = np.mean(stab2_vec), np.std(stab2_vec)

    print(algorithm.upper(), " over ", n_eval_episodes, " episodes")
    print("Return in perfect enviroment: ")
    print("mean: ", perf_mean, " standard deviation: ", perf_std)
    print("Max steady-state error norm in corrupted enviroment starting from setpoint: ")
    print("mean: ", stab1_mean, " standard deviation: ", stab1_std)
    print("Max steady-state error norm in corrupted enviroment starting outside of local domain: ")
    print("mean: ", stab2_mean, " standard deviation: ", stab2_std)
    perf_env.close()
    stab_env_1.close()
    stab_env_2.close()
