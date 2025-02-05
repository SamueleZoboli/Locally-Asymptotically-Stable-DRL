from robot_bases import MJCFBasedRobot
import numpy as np

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class InvertedPendulum(MJCFBasedRobot):
    swingup = False

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_pendulum.xml', 'cart', action_dim=1, obs_dim=4)


    def robot_specific_reset(self, bullet_client, s0):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["slider"]
        if s0 is None:
            u = self.np_random.uniform(low=-.1, high=.1)
            self.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
            # self.j2.reset_current_position(0 if not self.swingup else 0.5 * 10 * u,  u)
        else:
            self.j1.reset_current_position(s0[1], s0[3])
            self.j2.reset_current_position(s0[0], s0[2])
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        self.slider.set_motor_torque(100 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()

        x, vx = self.slider.current_position()
        assert (np.isfinite(x))
        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([x, angle_normalize(self.theta), vx, theta_dot])  # np.cos(self.theta), np.sin(self.theta)


class InvertedPendulumSwingup(InvertedPendulum):
    swingup = True


class InvertedDoublePendulum(MJCFBasedRobot):

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_double_pendulum.xml', 'cart', action_dim=1, obs_dim=6)

    def robot_specific_reset(self, bullet_client, s0):
        self._p = bullet_client
        self.pole2 = self.parts["pole2"]
        self.pole1 = self.parts["pole"]
        self.cart = self.parts["cart"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        body_idx=self.pole1.bodyIndex

        # set masses


        self._p.changeDynamics(self.pole1.bodyIndex, self.pole1.bodyPartIndex, mass=1)
        self._p.changeDynamics(self.pole2.bodyIndex, self.pole2.bodyPartIndex, mass=1)
        self._p.changeDynamics(self.slider.bodyIndex, self.cart.bodyPartIndex, mass=10)


        if s0 is None:
            u = self.np_random.uniform(low=-.1, high=.1)
            self.j1.reset_current_position(np.pi + u, 0)
            self.j2.reset_current_position(u, 0)
        else:
            self.slider.reset_current_position(s0[0], s0[3])
            self.j1.reset_current_position(s0[1], s0[4])
            self.j2.reset_current_position(s0[2], s0[5])
        self.j1.set_motor_torque(0)
        self.j2.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.slider.set_motor_torque(200 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()
        x, vx = self.slider.current_position()
        self.pos_x, _, self.pos_y = self.pole2.pose().xyz()
        assert (np.isfinite(x))
        return np.array([
            x,
            angle_normalize(theta),
            angle_normalize(gamma),
            vx,
            theta_dot,
            gamma_dot,
        ])


 ###############################################################################################################""
class InvertedWrongPendulumSwingup(MJCFBasedRobot):
    swingup = True
	
    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_wrong_pendulum.xml', 'cart', action_dim=1, obs_dim=4)
        

    def robot_specific_reset(self, bullet_client, s0):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["slider"]

        if s0 is None:
            u = self.np_random.uniform(low=-.1, high=.1)
            self.j1.reset_current_position(u if not swingup else 3.1415 + u, 0)
        else:
            self.j1.reset_current_position(s0[1], s0[3])
            self.j2.reset_current_position(s0[0], s0[2])
        self.j1.set_motor_torque(0)

    def apply_action(self, a, noise=False, steps=None):
        assert (np.isfinite(a).all())
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        if not noise:
            self.slider.set_motor_torque(100 * float(np.clip(a[0], -1, +1)))
        else:
            self.slider.set_motor_torque(
                100 * (float(np.clip(a[0], -1, +1)) + float(0.2 * np.sin(2 * np.pi * steps/ 50))))

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()
        x, vx = self.slider.current_position()
        assert (np.isfinite(x))

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([x, angle_normalize(self.theta), vx, theta_dot])  

class InvertedWrongDoublePendulum(MJCFBasedRobot):

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_wrong_double_pendulum.xml', 'cart', action_dim=1, obs_dim=6)

    def robot_specific_reset(self, bullet_client, s0):
        self._p = bullet_client
        self.pole2 = self.parts["pole2"]
        self.pole1 = self.parts["pole"]
        self.cart = self.parts["cart"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        body_idx = self.pole1.bodyIndex

        # set masses


        self._p.changeDynamics(self.pole1.bodyIndex, self.pole1.bodyPartIndex, mass=1)
        self._p.changeDynamics(self.pole2.bodyIndex, self.pole2.bodyPartIndex, mass=1.2)
        self._p.changeDynamics(self.slider.bodyIndex, self.cart.bodyPartIndex, mass=10)


        if s0 is None:
            u = self.np_random.uniform(low=-.1, high=.1)
            self.j1.reset_current_position(np.pi + u, 0)
            self.j2.reset_current_position(u, 0)
        else:
            self.slider.reset_current_position(s0[0], s0[3])
            self.j1.reset_current_position(s0[1], s0[4])
            self.j2.reset_current_position(s0[2], s0[5])
        self.j1.set_motor_torque(0)
        self.j2.set_motor_torque(0)

    def apply_action(self, a, noise=False, steps=None):
        assert (np.isfinite(a).all())
        if not noise:
            self.slider.set_motor_torque(200 * float(np.clip(a[0], -1, +1)))
        else:
            self.slider.set_motor_torque(
                200 * (float(np.clip(a[0], -1, +1)) + float(0.1 * np.sin(2 * np.pi * steps/ 100))))

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()
        x, vx = self.slider.current_position()
        self.pos_x, _, self.pos_y = self.pole2.pose().xyz()
        assert (np.isfinite(x))
        return np.array([
            x,
            angle_normalize(theta),
            angle_normalize(gamma),
            vx,
            theta_dot,
            gamma_dot,
        ])


