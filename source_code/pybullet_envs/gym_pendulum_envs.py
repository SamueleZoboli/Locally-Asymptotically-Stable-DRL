from .scene_abstract import SingleRobotEmptyScene
from .env_bases import MJCFBaseBulletEnv
from robot_pendula import InvertedPendulum, InvertedPendulumSwingup, InvertedDoublePendulum, \
    InvertedWrongPendulumSwingup, InvertedWrongDoublePendulum
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import os, sys


class InvertedPendulumBulletEnv(MJCFBaseBulletEnv):

    def __init__(self):
        self.robot = InvertedPendulum()
        self.origin_stop = False
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def set_origin_stop(self, stop=False):
        self.origin_stop = stop
        return self.origin_stop

    def reset(self, s0=None):
        if (self.stateId >= 0):
            # print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            self._p.restoreState(self.stateId)
        r = MJCFBaseBulletEnv.reset(self, s0).astype(self.observation_space.dtype)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            # print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state().astype(self.observation_space.dtype)  # sets self.pos_x self.pos_y
        # e=np.array([state[0], np.arctan2(state[2],state[1]), state[3], state[4]])
        # vel_penalty = 0
        if self.robot.swingup:
            x_cost = np.expand_dims(state, axis=0) @ self.Q @ np.expand_dims(state, axis=1)  # setpoint = origin
            u_cost = self.R * a ** 2  # eq_input = 0
            cost = x_cost.item() + u_cost.item()
            if self.origin_stop:
                done = True if (np.linalg.norm(state) < 1e-5) else False
            else:
                done = False  # not notdone
            reward = -cost
            # reward=np.cos(self.robot.theta)
            # done = False
        else:
            reward = 1.0
            done = np.abs(self.robot.theta) > .2
        self.rewards = [float(reward)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1, 0, 0, 0)


class InvertedPendulumSwingupBulletEnv(InvertedPendulumBulletEnv):

    def __init__(self):
        self.robot = InvertedPendulumSwingup()
        self.origin_stop = False
        self.Q = np.array([[0.15642212, 0.05884862, 0.02345707, 0.01208435],
                           [0.05884862, 0.23832731, 0.0420158, 0.02411947],
                           [0.02345707, 0.0420158, 0.0260213, 0.00847638],
                           [0.01208435, 0.02411947, 0.00847638, 0.01490583]])

        self.R = 0.0099
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1


class InvertedDoublePendulumBulletEnv(MJCFBaseBulletEnv):

    def __init__(self):
        self.robot = InvertedDoublePendulum()
        self.origin_stop = False
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        self.Q = np.array([[0.00265126, 0.009268, 0.0102381, 0.00191334, 0.00325421, 0.00194488],
                           [0.009268, 0.50915067, 0.26985838, 0.02084164, 0.0644325, 0.04627631],
                           [0.0102381, 0.26985838, 0.79271049, 0.02430241, 0.1015087, 0.07681595],
                           [0.00191334, 0.02084164, 0.02430241, 0.00513332, 0.00745765, 0.00457172],
                           [0.00325421, 0.0644325, 0.1015087, 0.00745765, 0.03391694, 0.01730611],
                           [0.00194488, 0.04627631, 0.07681595, 0.00457172, 0.01730611, 0.02290067]])
        self.R = 9.9e-5
        

    def set_origin_stop(self, stop=False):
        self.origin_stop = stop
        return self.origin_stop

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self, s0=None):
        if (self.stateId >= 0):
            self._p.restoreState(self.stateId)
        r = MJCFBaseBulletEnv.reset(self, s0).astype(self.observation_space.dtype)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state().astype(self.observation_space.dtype)  # sets self.pos_x self.pos_y
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
        # dist_penalty = 0.01 * self.robot.pos_x ** 2 + (self.robot.pos_y + 0.3 - 2) ** 2
        # # v1, v2 = self.model.data.qvel[1:3]   TODO when this fixed https://github.com/bulletphysics/bullet3/issues/1040
        # # vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        # vel_penalty = 0
        # alive_bonus = 10
        # done = self.robot.pos_y + 0.3 <= -1
        # self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
        x_cost = np.expand_dims(state, axis=0) @ self.Q @ np.expand_dims(state, axis=1)
        # theta, theta_dot = self.robot.j1.current_position()
        # gamma, gamma_dot = self.robot.j2.current_position()
        # x_cost=-np.cos(theta)-0.2*np.cos(gamma)
        u_cost = self.R * a ** 2
        cost = x_cost.item() + u_cost.item()
        if self.origin_stop:
            done = True if (np.linalg.norm(state) < 1e-4) else False
        else:
            done = False  # bool(y <= 1)
        self.rewards = [float(-cost)]
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.2, 0, 0, 0.5)


##################################################################################################################
#                               Noisy/added envs
##################################################################################################################

class InvertedPendulumSwingupNoisyBulletEnv(MJCFBaseBulletEnv):

    def __init__(self):
        self.robot = InvertedWrongPendulumSwingup()
        self.start_from_eq = False
        self.steps = 0
        self.worst = 0.
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def set_start_from_eq(self, eq=False):
        self.start_from_eq = eq
        return self.start_from_eq

    def reset(self, s0=None):
        if (self.stateId >= 0):
            self._p.restoreState(self.stateId)
        if self.start_from_eq:
            state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        else:
            state = np.array([0., np.pi, 0., 0.], dtype=self.observation_space.dtype)
        r = MJCFBaseBulletEnv.reset(self, state).astype(self.observation_space.dtype)
        self.steps = 0
        self.worst = 0
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
        return r

    def step(self, a):
        self.robot.apply_action(a, noise=True, steps=self.steps)
        self.scene.global_step()
        state = self.robot.calc_state().astype(self.observation_space.dtype)  # sets self.pos_x self.pos_y
        if self.steps < 999:
            cost = 0
            if self.steps > 500:  # look at max error norm at steady state
                error_norm = np.linalg.norm(state)
                self.worst = np.maximum(self.worst, error_norm)
        else:
            cost = self.worst
        done = False
        self.steps += 1
        self.rewards = [float(cost)]
        obs_noise = 0.03 * np.random.randn(self.observation_space.shape[0]).astype(self.observation_space.dtype)
        self.HUD(state, a, done)
        return state + obs_noise, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1, 0, 0, 0)


class InvertedDoublePendulumNoisyBulletEnv(MJCFBaseBulletEnv):

    def __init__(self):
        self.robot = InvertedWrongDoublePendulum()
        self.start_from_eq = False
        self.steps = 0
        self.worst = 0.
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def set_start_from_eq(self, eq=False):
        self.start_from_eq = eq
        return self.start_from_eq

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self, s0=None):
        if (self.stateId >= 0):
            self._p.restoreState(self.stateId)
        if self.start_from_eq:
            state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        else:
            state = np.array([0., np.pi, 0., 0., 0., 0.], dtype=self.observation_space.dtype)
        r = MJCFBaseBulletEnv.reset(self, state).astype(self.observation_space.dtype)
        self.steps = 0
        self.worst = 0
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
        return r

    def step(self, a):
        self.robot.apply_action(a, noise=True, steps=self.steps)
        self.scene.global_step()
        state = self.robot.calc_state().astype(self.observation_space.dtype)  # sets self.pos_x self.pos_y
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
        if self.steps < 1999:
            cost = 0
            if self.steps > 1500:  # look at max error norm at steady state
                error_norm = np.linalg.norm(state)
                self.worst = np.maximum(self.worst, error_norm)
        else:
            cost = self.worst
            # print(cost)
        done = False
        self.steps += 1
        self.rewards = [float(cost)]
        obs_noise = 0.03 * np.random.randn(self.observation_space.shape[0]).astype(self.observation_space.dtype)
        self.HUD(state, a, done)
        return state + obs_noise, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.2, 0, 0, 0.5)



