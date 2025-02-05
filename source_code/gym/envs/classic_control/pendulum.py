import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=9.81):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1
        self.viewer = None

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.Q = np.array([[1.06336303, 0.00130376],
                           [0.00130376, 0.10034978]])
        self.R = 0.00099

        self.origin_stop = False

        self.x_star = np.array([10 / 180 * np.pi, 0.], dtype=np.float32)
        self.u_bar = -0.85174636

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_origin_stop(self, stop=False):
        self.origin_stop = stop
        return self.origin_stop

    def step(self, actions):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(actions, -self.max_torque, self.max_torque)[0]  
        self.last_u = u  # for rendering
        x = np.array([[th - self.x_star[0]], [thdot - self.x_star[1]]])
        x_cost = (x.T @ self.Q @ x).item()
        u_cost = self.R * (u - self.u_bar) ** 2
        costs = x_cost + u_cost  # angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = angle_normalize(newth)

        self.state = np.array([newth, newthdot])

        if self.origin_stop:
            done = True if (np.linalg.norm(self.state - self.x_star) < 1e-5) else False
        else:
            done = False

        return self._get_obs(), -costs, done, {}

    def reset(self, s0=None):
        if s0 is not None:
            self.state = np.array([angle_normalize(s0[0]), np.clip(s0[1], -self.max_speed, self.max_speed)],
                                  dtype=np.float32)
        else:
            high = np.array([np.pi, 1.], dtype=np.float32)
            self.state = self.np_random.uniform(low=-high, high=high).astype(high.dtype)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot], dtype=np.float32)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
