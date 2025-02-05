import scipy.signal
from scipy import linalg
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from typing import Dict, List
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.torch_layers import create_mlp
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys


def weighted_mse(x, y, w):
    mse = (x-y)**2
    out = w * mse
    loss = out.mean()
    return loss


class LocalController():
    """Local controller object, computes the linear model and the LQR """

    def __init__(self,
                 vec_env: VecEnv,
                 gamma: float,
                 loadparams: Dict = None):
        """
        :param vec_env: Environmet to act on
        :param gamma: Discount factor
        :param loadparams: Dictionary of parameters to be loaded (doa,P, H, K, ubar,xstar) if loading a model"""
        if loadparams is None:
            env = vec_env.envs[0]  # work with a single env
            env = env.unwrapped
            env.set_origin_stop(False)
            env.reset()
            self.u_bar = self.get_linear_matrices(env)
            K, P, H = self.compute_feedback(gamma=gamma, env=env)
            doa = self.estimate_doa(env, self.u_bar, self.x_star, K, P, 0.01, 5000, 0.3)
            self.doa = th.tensor(doa)
            self.P = th.from_numpy(P)
            self.H = th.from_numpy(H.astype(env.action_space.dtype))
            self.K = th.from_numpy(K)
            self.u_bar = th.from_numpy(self.u_bar.astype(env.action_space.dtype))
            self.x_star = th.from_numpy(self.x_star.astype(env.observation_space.dtype))
            self.doa.requires_grad = False
            self.P.requires_grad = False
            self.H.requires_grad = False
            self.K.requires_grad = False
            self.u_bar.requires_grad = False
            self.x_star.requires_grad = False
            env.set_origin_stop(True)
        else:
            self.doa = loadparams['doa']
            self.P = loadparams['p']
            self.H = loadparams['h']
            self.K = loadparams['k']
            self.u_bar = loadparams['ubar']
            self.x_star = loadparams['xs']

    def get_linear_matrices(self, env: GymEnv):
        """Obtain linear system matrices and feedforward action if necessary
        :param env: Environment to linearize"""

        # Env info
        name = env.spec.id
        self.n = env.observation_space.shape[0]
        self.m = env.action_space.shape[0]

        if name == 'Pendulum-v0':
            # cost matrices
            self.Q = np.array([[1, 0],
                               [0, 0.1]])
            self.R = np.array([[0.001]])
            # setpoint
            self.x_star = np.array([10 / 180 * np.pi, 0.])
            # linearization box bounds
            self.x_min = np.array([-15 / 180 * np.pi, -5 / 180 * np.pi])
            self.x_max = np.array([15 / 180 * np.pi, 5 / 180 * np.pi])


        elif name == 'InvertedPendulumSwingupBulletEnv-v0':
            # cost matrices
            self.Q = np.array([[0.1, 0, 0, 0],
                               [0, 0.1, 0, 0],
                               [0, 0, 0.01, 0],
                               [0, 0, 0, 0.01]])
            self.R = np.array([[0.01]])
            # setpoint
            self.x_star = np.zeros(env.observation_space.shape[0], dtype=env.observation_space.dtype)  # x,th, dx, dth
            # linearization box bounds
            self.x_min = np.array([-.1, -10 / 180 * np.pi, -.1, -2 / 180 * np.pi], dtype=env.observation_space.dtype)
            self.x_max = np.array([.1, 10 / 180 * np.pi, .1, 2 / 180 * np.pi], dtype=env.observation_space.dtype)


        elif name == 'InvertedDoublePendulumBulletEnv-v0':
            # cost matrices
            self.Q = np.array([[0.001, 0, 0, 0, 0, 0],
                               [0, 0.3, 0, 0, 0, 0],
                               [0, 0, 0.3, 0, 0, 0],
                               [0, 0, 0, 0.001, 0, 0],
                               [0, 0, 0, 0, 0.01, 0],
                               [0, 0, 0, 0, 0, 0.01]])
            self.R = np.array([[0.0001]])
            # setpoint
            self.x_star = np.zeros(env.observation_space.shape[0], dtype=env.observation_space.dtype)
            # linearization box bounds
            self.x_min = np.array([-.1, -5 / 180 * np.pi, -5 / 180 * np.pi, -.1, -5 / 180 * np.pi, -5 / 180 * np.pi])
            self.x_max = np.array([.1, 5 / 180 * np.pi, 5 / 180 * np.pi, .1, 5 / 180 * np.pi, 5 / 180 * np.pi])


        # Compute feedforward action with genetic algorithm and gradient descent
        u_bar = self.feedforward_action(env, self.x_star)
        # identify linear model with least squares
        self.A, self.B, self.C, self.D = self.identify(env, self.x_star, u_bar, self.x_min, self.x_max, self.n, self.m,
                                                       int(3e5))
        self.res_id_mean, self.res_id_std, self.res_id_corr = self.validate(
            env, u_bar, self.x_star, self.x_min, self.x_max, int(5e4), self.A, self.B, self.n, self.m)
        print('\n Linear model stats: ')
        print(
            ' residual mean: %s \n residual standard dev: %s \n residual corr: %s' % (
                self.res_id_mean, self.res_id_std, self.res_id_corr))

        # Verify controllability of identified model
        Co = self.B
        for i in range(1, self.n):
            Co = np.append(Co, np.linalg.matrix_power(self.A, i) @ self.B, axis=1)
        assert np.linalg.matrix_rank(Co) == self.n, 'Uncontrollable linear system'

        return u_bar

    def feedforward_action(self, env: GymEnv, xs: np.ndarray,
                           ga_params: Dict = {'max_num_iteration': 3000, \
                                              'population_size': 100, \
                                              'mutation_probability': 0.1, \
                                              'elit_ratio': 0.01, \
                                              'crossover_probability': 0.5, \
                                              'parents_portion': 0.3, \
                                              'crossover_type': 'uniform', \
                                              'max_iteration_without_improv': None}):
        """Learn u(x_star) which maxes x_star an equilibrium
        :param env: The enviroment on which the net must be trained
        :param xs: The desired equilibrium state
        :param ga_params: Genetic algorithm parameters
        """
        x_goal = xs

        def fitness(u0): # distance from x_star after one step
            x0 = env.reset(s0=x_goal)
            xn, _, _, _ = env.step(u0)
            e = np.expand_dims(xn - x0, axis=1)
            Q = np.identity(e.shape[0])
            u = np.expand_dims(u0, axis=1)
            cost = (e.T @ Q @ e ).item()
            return cost

        def refine_grad_desc(v):
            #gradient descent parameters
            epsilon = 0.001 * env.action_space.high[0]
            alpha = 0.001
            # gradient escent
            v0 = v.copy()
            n_iter = 0
            max_iter = int(1e6)
            minim = False
            print('Gradient descent:')
            while fitness(v0) > 1e-12 and n_iter < max_iter and not minim:
                y0 = fitness(v0)
                y1k = np.zeros(v0.shape)
                y2k = np.zeros(v0.shape)
                for k in range(v0.shape[0]):
                    v1 = v0.copy()
                    v1[k] -= epsilon
                    y1k[k] = fitness(v1)
                    v2 = v0.copy()
                    v2[k] += epsilon
                    y2k[k] = fitness(v2)
                grad = (y2k - y1k) / (2 * epsilon)
                v0 -= alpha * grad
                y1 = fitness((v0))
                if np.absolute(y0 - y1) < 1e-20:
                    minim = True
                sys.stdout.write('\r Current cost: %s' % (str(fitness(v0))))
                sys.stdout.flush()
                n_iter += 1
            return v0

        # Test first if 0 input is stabilizing
        u0 = np.zeros((env.action_space.shape[0],))
        cost = fitness(u0)
        forward_needed = True if cost > 1e-8 else False

        if forward_needed:
            # Estimate the feedforward action via the genetic algorithm and refine the solution with gradient descent
            low, high = env.action_space.low, env.action_space.high
            f = fitness
            varbounds = np.concatenate((np.expand_dims(low, axis=1), np.expand_dims(high, axis=1)), axis=1)
            genetic = ga(function=f, dimension=env.action_space.shape[0], variable_type='real',
                         variable_boundaries=varbounds,
                         algorithm_parameters=ga_params, convergence_curve=False)
            genetic.run()
            u_bar = genetic.best_variable
            u_bar = refine_grad_desc(u_bar)
            x0 = env.reset(s0=x_goal)
            x1, _, _, _ = env.step(u_bar)
            np.set_printoptions(precision=8, suppress=True)
            print('\n Refined solution: %s' % (u_bar))
            print('\n One step error: %s' % (x1 - x0))
            assert np.all(np.absolute(x1 - x0) < 1e-3), 'No stabilizing feedforward  action found'
        else:
            u_bar = np.zeros((env.action_space.shape[0],))
            print('\n No feedforward action needed')

        return u_bar

    def identify(self, env: GymEnv, x_star: np.ndarray, u_bar: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, n: int,
                 m: int, nstps: int):
        """Linear system identification, collect local trajectories and use least squares
        :param env: Environment to be identified
        :param x_star: Equilibrium state
        :param u_bar: Feedforward input making x_star an equilibrium
        :param x_min: Minimum values for the state exploration set
        :param x_max: Maximum values for the state exploration set
        :param n: State vector size
        :param m: Input vector size
        :param nstps: Number of data to be collected"""
        N = nstps
        name = env.spec.id
        # Allocate memory
        y = np.zeros((N, n),
                     dtype=env.observation_space.dtype)  # x(t+1), first dim has to be batch for numpy broadcsting
        x = np.zeros((N, n), dtype=env.observation_space.dtype)

        # Initial condition
        x0 = x_star

        u = u_bar + 0.01 * env.action_space.high * np.random.randn(N, m).astype(
            env.action_space.dtype)  # N small actions

        x[0, :] = env.reset(s0=x0)

        # Generate data
        for i in range(N):
            action = u[i, :]
            obs, rew, done, _ = env.step(action)
            y[i, :] = obs
            if i < N - 1:
                if (np.any(y[i, :] < x_star + x_min) or np.any(y[i, :] > x_star + x_max)):
                    x0 = x_star  # new initial condition
                    x[i + 1, :] = env.reset(s0=x0)
                else:
                    x[i + 1, :] = y[i, :]

        Phi = np.concatenate((x, u), axis=1)  # state-input vector for ARX model (our linear model)
        params, res, ran, s = np.linalg.lstsq(Phi, y, rcond=None)
        params = params.T

        # linear model x_next = A x + B u, y = C x + D u, the output y is not actually used, we use the full state x
        A = params[:, 0:n]
        B = params[:, n:]
        C = np.identity(n)  # full state obs
        D = np.zeros((n, m))  # no input onto the output

        return A, B, C, D

    def validate(self, env: GymEnv, u_bar: np.ndarray, x_star: np.ndarray, x_min: np.ndarray, x_max: np.ndarray,
                 nstps: int, A: np.ndarray, B: np.ndarray, n: int, m: int):
        """Validate the linear model
        :param env: Environment to be identified
        :param u_bar: Feedforward input making x_star an equilibrium
        :param x_star: Equilibrium state
        :param x_min: Minimum values for the state exploration set
        :param x_max: Maximum values for the state exploration set
        :param nstps: Number of data to be collected
        :param A: Linear state matrix
        :param B: Linear input matrix
        :param n: State vector size
        :param m: Input vector size"""
        # Validate the model
        N = nstps
        name = env.spec.id
        params = np.concatenate((A, B), axis=1)
        # Allocate memory
        y = np.zeros((N, n), dtype=env.observation_space.dtype)  # x(t+1)
        x = np.zeros((N, n), dtype=env.observation_space.dtype)
        u = np.zeros((N, m), dtype=env.action_space.dtype)

        x0 = x_star + 0.3 * (x_max) * np.random.randn(n).astype(
            env.observation_space.dtype)  # random initial cond close to the equilibrium
        x[0, :] = env.reset(s0=x0)

        # Generate data
        for i in range(N):
            action = env.action_space.sample()
            u[i, :] = action

            obs, rew, done, _ = env.step(action)
            y[i, :] = obs

            if i < N - 1:
                if (np.any(y[i, :] < x_star + x_min) or np.any(y[i, :] > x_star + x_max)):
                    x0 = x_star + 0.3 * x_max * np.random.randn(n).astype(
                        env.observation_space.dtype)  # new initial condition
                    x[i + 1, :] = env.reset(s0=x0)
                else:
                    x[i + 1, :] = y[i, :]

        # Residual analysis
        Phi = np.concatenate((x, u), axis=1)
        res_id = y - Phi @ params.T
        mean_id = np.mean(res_id, axis=0)
        std_id = np.std(res_id, axis=0)

        corr_id = np.zeros((n,))
        for j in range(n):
            corr_id[j] = np.correlate(res_id[:, j], res_id[:, j])

        return mean_id, std_id, corr_id

    def compute_feedback(self, gamma: float, env: GymEnv):
        # Compute the linear feedback

        # Discounted LQR, the P is the same as the undiscounted one due to the choice of Q_gamma and R_famma
        P = linalg.solve_discrete_are(self.A , self.B , self.Q, self.R)  # undiscounted DARE
        assert np.all(np.linalg.eigvals(P) > 0), 'P is not positive definite'

        self.Q_gamma = gamma * self.Q + (1 - gamma) * P
        self.R_gamma = gamma * self.R

        # Controller
        M1 = linalg.inv(self.R_gamma + gamma * self.B.T @ P @ self.B)
        M2 = self.B.T @ P @ self.A
        K = -gamma * M1 @ M2  # optimal discounted LQR gain
        H11 = self.Q_gamma + gamma * self.A.T @ P @ self.A
        H12 = gamma * self.A.T @ P @ self.B
        H21 = gamma * self.B.T @ P @ self.A
        H22 = self.R_gamma + gamma * self.B.T @ P @ self.B
        H1 = np.concatenate((H11, H12), axis=1)
        H2 = np.concatenate((H21, H22), axis=1)
        H = np.concatenate((H1, H2), axis=0)

        print('Linear system cl-eig: ', np.linalg.eigvals((self.A + self.B @ K)))

        return K, P, H

    def estimate_doa(self, env: GymEnv, u_bar: np.ndarray, x_star: np.ndarray, gain: np.ndarray, dare: np.ndarray,
                     step: float, npts: int, perc_fail: float):
        """Estimate the domain of attraction of the local controller with the linear system Lyapunov function ellipsoid
        :param env: Environment where to test the controller
        :param u_bar: Feedforward term
        :param x_star: Equilibrium state
        :param gain: Feedback gain matrix
        :param dare: Solution of the DARE
        :param step: Step size between level sets
        :param npts: Number of samples per level set
        :param perc_fail: Percentage of failed Lyap condition verifications per level set for considering the limits of the domain of attraction"""
        a_min = env.action_space.low
        a_max = env.action_space.high
        c = step
        n_points = npts
        percent_fail = perc_fail
        P = dare.astype(env.observation_space.dtype)
        K = gain.astype(env.observation_space.dtype)
        n = P.shape[0]
        done = False

        while not done:
            # Random samples on the n-dimensional ellipsoid surface x^T P x = c
            L = np.linalg.cholesky(P)  # lower triangular
            rv = np.random.randn(n_points + int(n_points * 0.01 * c / step),
                                 n)  # an array of n_points normally distributed n_dimensional random variables
            norm = np.sqrt(np.sum(rv ** 2, axis=1, keepdims=True))
            x0s = np.sqrt(c) * rv / norm  # uniform sample in sqrt(c)-radius n-dimensional sphere
            x0s = np.linalg.inv(L.T) @ np.expand_dims(x0s, axis=2)
            x0s = x0s.squeeze(2) + x_star
            max_fails = int((n_points + n_points * 0.01 * c / step) * percent_fail)

            # for each random initial condition test LQR
            n_fail = 0
            for i in range(x0s.shape[0]):
                s0 = x0s[i, :].astype(env.observation_space.dtype)
                obs = env.reset(s0=s0)
                done = False
                x = np.expand_dims(obs - x_star, axis=1)
                xT = np.expand_dims(obs - x_star, axis=0)
                V_x0 = (xT @ P @ x).item()
                u = u_bar + np.squeeze((K @ x), axis=1)
                action = np.clip(u, a_min, a_max)  # saturate action
                obs, _, _, _ = env.step(action)
                x = np.expand_dims(obs - x_star, axis=1)
                xT = np.expand_dims(obs - x_star, axis=0)
                V_x1 = (xT @ P @ x).item()
                if V_x1 - V_x0 > 0:  # Lyapunov decrease check, x^T P x is a local Lyap f(x)
                    n_fail += 1
                    if n_fail > max_fails:
                        doa = c
                        done = True
                        break
            c = c + step
        print('Estimated DOA= %.5f' % doa)

        return doa

    def act(self, obs: th.Tensor):
        """Compute the local controller action
        :param obs: State vector"""
        x = (obs - self.x_star).unsqueeze(2).type(
            self.K.dtype)  # x must be a (batch x n x 1) for correct matrix computations
        xT = (obs - self.x_star).unsqueeze(1).type(self.K.dtype)  # transpose accounting for batch dim
        u = self.u_bar + (self.K @ x).squeeze(2)
        return u

    def evaluate(self, obs: th.Tensor, act: th.Tensor = None):
        """Compute the local framework value
        :param obs: State vector
        :param act: Input vector"""
        x = (obs - self.x_star).unsqueeze(2).type(self.P.dtype)
        xT = (obs - self.x_star).unsqueeze(1).type(self.P.dtype)
        if act is None:  # state-value function
            v = (xT @ self.P @ x).squeeze(2)
            return v.type(obs.dtype)
        else:  # action-value function
            u = (act - self.u_bar).unsqueeze(2)
            uT = (act - self.u_bar).unsqueeze(1)
            z = th.cat((x, u), dim=1).type(self.H.dtype)  # get a (batch x n+m x 1) tensor
            zT = th.cat((xT, uT), dim=2).type(self.H.dtype)
            q = (zT @ self.H @ z).squeeze(2)
            return q.type(act.dtype)

    def smooth_sat(self, obs: th.Tensor, alpha: float, order: int):
        """Compute the smooth saturation of a specific order at the point
        :param obs: State vector where to evaluate the saturation
        :param alpha: Value of the saturation at the limits of the estimated DOA, 0<alpha<1
        :param order: The order of the saturation function"""

        x = (obs - self.x_star).unsqueeze(2).type(self.P.dtype)
        xT = (obs - self.x_star).unsqueeze(1).type(self.P.dtype)
        beta = th.atanh(th.tensor(alpha)).item()
        if order == 2:
            h_x = th.tanh(beta * (xT @ self.P @ x).squeeze(2) / self.doa)
        else:
            h_x = th.tanh(beta * th.pow(th.sqrt((xT @ self.P @ x).squeeze(2) / self.doa), order))

        return h_x

    def to(self, device):
        self.doa = self.doa.to(device)
        self.P = self.P.to(device)
        self.H = self.H.to(device)
        self.K = self.K.to(device)
        self.u_bar = self.u_bar.to(device)
        self.x_star = self.x_star.to(device)

