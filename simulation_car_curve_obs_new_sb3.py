import math
import numpy as np
import new_env_parameters_setting_sb3 as env_para

class Simulation:

    def __init__(self, dt, goal_pos):
        env = env_para.parameters()
        self.FIELD_SIZE_x_low = env.FIELD_SIZE_x_low
        self.FIELD_SIZE_x_up = env.FIELD_SIZE_x_up
        self.FIELD_SIZE_y_low = env.FIELD_SIZE_y_low
        self.FIELD_SIZE_y_up = env.FIELD_SIZE_y_up
        self._state = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.next_state = np.array([0., 0., 0., 0., 0., 0., 0.])

        # time
        self.ticks = 0
        self.dt = dt
        # self.dθ = math.pi / 30
        self.spin = 0
        self.L = env.L    # 车辆轴距
        self.goal_pos = goal_pos
        self.FIELD_SIZE_x_low = env.FIELD_SIZE_x_low
        self.FIELD_SIZE_x_up = env.FIELD_SIZE_x_up
        self.FIELD_SIZE_y_low = env.FIELD_SIZE_y_low
        self.FIELD_SIZE_y_up = env.FIELD_SIZE_y_up
        self.MIN_ACC = env.MIN_ACC
        self.MAX_ACC = env.MAX_ACC
        self.MAX_omiga = env.MAX_omiga
        self.theta_low = env.theta_low
        self.theta_high = env.theta_high
        self.v_low = env.v_low
        self.v_high = env.v_high
        self.phi_low = env.phi_low
        self.phi_high = env.phi_high
        self.goal_dist_low = env.goal_dist_low
        self.goal_dist_high = env.goal_dist_high

    @property
    def time(self):
        return round(self.ticks * self.dt, 4)

    def is_invalid(self):
        """
        Check if out of bounds.
        """
        _, _, x, y, _, _, _ = self.next_state  # 5-state
        return x < self.FIELD_SIZE_x_low or x > self.FIELD_SIZE_x_up or y < self.FIELD_SIZE_y_low or y > self.FIELD_SIZE_y_up

    @property
    def speed(self):
        return self.next_state[[0, 1]]

    @property
    def position(self):
        return self.next_state[[2, 3]]

    @property
    def theta(self):
        return self.next_state[-3]

    @property
    def a(self):
        return self.next_state[0]

    @property
    def w(self):
        return self.next_state[1]

    @property                                 # 3-state
    def v(self):
        return self.next_state[-2]

    @property                                 # 3-state
    def phi(self):
        return self.next_state[-1]

    @property
    def action(self):
        return self.next_state[0:2]

    def step(self, action):
        next_a, next_w = action
        self._state = self.next_state.copy()
        a, w, x, y, theta, v, phi = self._state

        # Update state
        new_v = v + next_a * self.dt
        if new_v > self.v_high:
            new_v = self.v_high
        if new_v < self.v_low:
            new_v = self.v_low
        #
        new_phi = phi + next_w * self.dt
        if new_phi > self.phi_high:
            new_phi = self.phi_high
        if new_phi < self.phi_low:
            new_phi = self.phi_low

        new_theta = theta + (new_v * np.tan(new_phi) / self.L * self.dt)

        if new_theta > np.pi:
            new_theta -= 2 * np.pi
        elif new_theta < -np.pi:
            new_theta += 2 * np.pi

        noise = np.random.normal(0, 1e-6)
        new_x = x + new_v * np.cos(new_theta) * self.dt + noise
        new_y = y + new_v * np.sin(new_theta) * self.dt + noise

        self.ticks += 1
        self.next_state = np.array([next_a, next_w, new_x, new_y, new_theta, new_v, new_phi])

    def reset(self, ep, intial_diy=False,fixed=False,pos_ini=None):

        env_para_ = env_para.parameters()
        self.obstacle_origin = env_para_.obstacle
        self.obstacle_no = env_para_.obstacle_no
        self.obstacle = np.zeros((2, len(self.obstacle_origin[0, :])))
        self.obstacle_random = np.zeros((2, len(env_para_.r)))

        self.r = env_para_.r
        self.obs_num = len(self.obstacle[0,:])

        self.R = env_para_.R

        if fixed is True:
            if pos_ini == None:

                rand_x = 7.7 + np.random.uniform(-1, 1)
                rand_y = 7 + np.random.uniform(-1, 1)
            else:
                print('pos_ini:',pos_ini)
                rand_x = np.array(pos_ini[0][0][0],dtype=np.float32)
                print('pos_ini_X:',pos_ini[0][0][0])
                rand_y = np.array(pos_ini[0][1][0],dtype=np.float32)
                print('pos_ini_Y:',rand_y)
        else:
            k = 1
            rand_x = np.random.uniform(4, 7.8) / k
            rand_y = np.random.uniform(4, 7.8) / k
            pos = np.array([rand_x, rand_y])
            goal_dist = np.linalg.norm(pos - self.goal_pos)
            obs_dist = np.linalg.norm(np.array([pos]).T - self.obstacle_random, axis=0)
            while  goal_dist < 5 or np.any(obs_dist<1.8):
                rand_x = np.random.uniform(3, 7.8) / k
                rand_y = np.random.uniform(3, 7.8) / k
                pi = np.array([rand_x, rand_y])
                goal_dist = np.linalg.norm(pi - self.goal_pos)
                obs_dist = np.linalg.norm(np.array([pi]).T - self.obstacle_origin,axis=0)


        self.ticks = 0
        rand_theta = np.arctan2(self.goal_pos[1] - rand_y, self.goal_pos[0] - rand_x) + np.random.uniform(-np.pi/10, np.pi/10)

        if rand_theta > np.pi:
            rand_theta -= 2*np.pi
        elif rand_theta < -np.pi:
            rand_theta += 2*np.pi
        rand_φ = 0

        if intial_diy is False:   # abandon
            self.start_point = np.array([rand_x, rand_y])
            self._state = np.array([
                0,
                0,
                rand_x, #小车真正的初始位置x
                rand_y, #小车位置y
                rand_theta,
                0,
                rand_φ
            ])
            self.next_state = np.array([
                0,
                0,
                rand_x, #小车真正的初始位置x
                rand_y, #小车位置y
                rand_theta,
                0,
                rand_φ
            ])
            self.start_state = self._state.copy()

        else:
                if ep < 30:
                    self.start_point = np.array([rand_x, rand_y])
                    self._state = np.array([
                        0.,
                        0.,
                        rand_x,
                        rand_y,
                        np.pi,
                        0,
                        0
                    ])
                    self.next_state = np.array([
                        0.,
                        0.,
                        rand_x,
                        rand_y,
                        np.pi,
                        0,
                        0
                    ])
                    self.start_state = self._state.copy()
                elif ep < 900 and ep >= 30:
                    k = 3
                    rand_x2 = np.random.uniform(self.FIELD_SIZE_x_low, self.FIELD_SIZE_x_up) / k
                    rand_y2 = np.random.uniform(self.FIELD_SIZE_y_low, self.FIELD_SIZE_y_up) / k
                    self.start_point = np.array([rand_x2, rand_y2])
                    rand_theta2 = np.arctan2(self.goal_pos[1] - rand_y2, self.goal_pos[0] - rand_x2)
                    self._state = np.array([
                        0,
                        0,
                        rand_x2,
                        rand_y2,
                        rand_theta2,
                        0.5,
                        0
                    ])
                    self.next_state = np.array([
                        0,
                        0,
                        rand_x2,
                        rand_y2,
                        rand_theta2,
                        0.5,
                        0
                    ])
                    self.start_state = self._state.copy()
                elif ep < 1500 and ep >= 900:
                    k = 2
                    rand_x3 = np.random.uniform(self.FIELD_SIZE_x_low, self.FIELD_SIZE_x_up) / k
                    rand_y3 = np.random.uniform(self.FIELD_SIZE_y_low, self.FIELD_SIZE_y_up) / k
                    self.start_point = np.array([rand_x3, rand_y3])
                    # rand_theta3 = np.arctan2(self.goal_pos[1] - rand_y3, self.goal_pos[0] - rand_x3)
                    rand_theta3 = np.random.uniform(0, 2 * np.pi)
                    self._state = np.array([
                        0,
                        0,
                        rand_x3,
                        rand_y3,
                        rand_theta3,
                        1,
                        0
                    ])
                    self.next_state = np.array([
                        0,
                        0,
                        rand_x3,
                        rand_y3,
                        rand_theta3,
                        1,
                        0
                    ])
                    self.start_state = self._state.copy()
                else:
                    k = 1
                    rand_x_ = np.random.uniform(self.FIELD_SIZE_x_low, self.FIELD_SIZE_x_up) / k
                    rand_y_ = np.random.uniform(self.FIELD_SIZE_y_low, self.FIELD_SIZE_y_up) / k
                    self.start_point = np.array([rand_x, rand_y])
                    rand_theta_ = np.random.uniform(0, 2*np.pi)
                    self._state = np.array([
                        0,
                        0,
                        rand_x_,
                        rand_y_,
                        rand_theta_,
                        0,
                        0
                    ])
                    self.next_state = np.array([
                        0,
                        0,
                        rand_x_,
                        rand_y_,
                        rand_theta_,
                        1,
                        0
                    ])
                    self.start_state = self._state.copy()


    def _get_dist(self, p1: np.ndarray, p2: np.ndarray):
        return np.linalg.norm(p1 - p2)

    def set_orig(self,x,y):
        self._state[2] = x
        self._state[3] = y

    def reset_mode(self, itr, goal_pos, mode='gradual', ):
        self.ticks = 0
        self.goal_pos = goal_pos
        rand_x = 35
        rand_y = 20
        # ensure orig theta
        self.start_point = np.array([rand_x,rand_y])
        rand_theta = np.pi
        # rand_φ = np.random.uniform(-np.pi / 3, np.pi / 3) np.random.uniform(3*np.pi/4, np.pi/4)
        rand_φ = 0
        self._state = np.array([
            0,
            0,
            rand_x, #小车真正的初始位置x
            rand_y, #小车位置y
            rand_theta,
            0,
            rand_φ
        ])
        self.start_state = self._state.copy()
        
