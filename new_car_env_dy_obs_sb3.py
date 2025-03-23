import numpy as np
import gymnasium as gym
from simulation_car_curve_obs_new_sb3 import Simulation
from viewer_new import Viewer
from implicit_policy import load_V, ds_stabilizer
import new_env_parameters_setting_sb3 as env_para
from gymnasium import spaces
from typing import Optional



REWARD_LIMIT = 500


# car_env_demo = CarEnv_demo()

class CarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        super(CarEnv, self).__init__()

        # self.random_obs = False
        self.random_obs = True
        # self.train_num = 5000000
        self.step_num = 0.

        # Time delta per step
        self.dt = 0.01

        # Environment parameters
        env_para_ = env_para.parameters()
        self.obstacle_origin = env_para_.obstacle
        self.obstacle_no = env_para_.obstacle_no
        self.obstacle = np.zeros((2, len(self.obstacle_origin[0, :])))
        self.obstacle_random = np.zeros((2, len(env_para_.r_rand)))

        self.r = env_para_.r
        self.r_rand = env_para_.r_rand
        self.obs_num = len(self.obstacle[0, :])

        self.R = env_para_.R

        self.goal_pos = env_para_.goal_pos

        self.sim = Simulation(self.dt, self.goal_pos)
        self.get_virtual_position = lambda: self.sim.position
        self.get_virtual_theta = lambda: self.sim.theta
        # self.seed(2)
        self.ep = 0.0

        # Boundaries of the action
        self.MIN_ACC = env_para_.MIN_ACC
        self.MAX_ACC = env_para_.MAX_ACC
        self.MAX_omiga = env_para_.MAX_omiga
        self.u_upbound = env_para_.u_uobound
        self.u_lowbound = env_para_.u_lowbound
        self.is_discrete_action = False

        # how close to goal = reach goal
        # self.dist_threshold = 0.01
        self.dist_threshold = env_para_.dist_threshold
        self.obs_theta_high = env_para_.obs_theta_high
        self.obs_theta_low = env_para_.obs_theta_low
        # how close to obstacle = crash obstacle
        self.obs_threshold = env_para_.obs_threshold
        self.obs_index = 0

        # Action and observation spaces
        # self.FIELD_SIZE = 40
        self.FIELD_SIZE_x_low = env_para_.FIELD_SIZE_x_low
        self.FIELD_SIZE_x_up = env_para_.FIELD_SIZE_x_up
        self.FIELD_SIZE_y_low = env_para_.FIELD_SIZE_y_low
        self.FIELD_SIZE_y_up = env_para_.FIELD_SIZE_y_up
        self.theta_low = env_para_.theta_low
        self.theta_high = env_para_.theta_high
        self.v_low = env_para_.v_low
        self.v_high = env_para_.v_high
        self.phi_low = env_para_.phi_low
        self.phi_high = env_para_.phi_high
        self.error_theta_low = env_para_.error_theta_low
        self.error_theta_high = env_para_.error_theta_high
        self.goal_dist_low = env_para_.goal_dist_low
        self.goal_dist_high = env_para_.goal_dist_high
        self.ux_low = env_para_.ux_low
        self.ux_high = env_para_.ux_high
        self.uy_low = env_para_.uy_low
        self.uy_high = env_para_.uy_high
        self.delta_v_rad_low = env_para_.delta_v_rad_low
        self.delta_v_rad_high = env_para_.delta_v_rad_high

        self.num_step = 0

        self.state_low = np.array(
            [self.goal_dist_low,
             self.FIELD_SIZE_x_low, self.FIELD_SIZE_y_low,
             self.theta_low,
             self.v_low,
             self.phi_low,
             0,
             self.delta_v_rad_low,
             self.MIN_ACC, -self.MAX_omiga
             ], dtype=np.float32)
        self.state_high = np.array(
            [self.goal_dist_high,

             self.FIELD_SIZE_x_up, self.FIELD_SIZE_y_up,
             self.theta_high,
             self.v_high,
             self.phi_high,
             self.obs_threshold,
             self.delta_v_rad_high,
             self.MAX_ACC, self.MAX_omiga
             # self.u_upbound,self.u_upbound,
             # 0
             ], dtype=np.float32)


        self.action_space = spaces.box.Box(
            low=np.array([self.MIN_ACC, -self.MAX_omiga], dtype=np.float32),
            high=np.array([self.MAX_ACC, self.MAX_omiga], dtype=np.float32),
            dtype=np.float32)


        self.observation_space = spaces.box.Box(
            low=self.state_low,
            high=self.state_high,
            dtype=np.float32)

        self.goal_dist = 10.
        self.delta_a = 0.
        self.delta_w = 0.
        self.a_real = 0.
        self.w_real = 0.
        self.a_ref = 0.
        self.w_ref = 0.
        self.a_ref_norm = 0.
        self.w_ref_norm = 0.
        self.u = np.zeros(2)
        self.dx_u = np.zeros(2)
        self.dx_car = np.zeros(2)
        self.dx_car_ = np.zeros(2)
        self.dx_car_theta = 0.
        self.delta_dudx_cos = 0.
        self.u_incar = np.zeros(2)
        self.dx_car_incar = np.zeros(2)
        self.Vx = np.zeros(2)
        self.Vdot = np.zeros(2)
        self.delta_v_rad = 0.
        self.delta_v_rad_last = 0.
        self.delta_mid_reward_V = 0.
        self.V = 20.
        self.u_count = 0
        self.V_state_scaled = 0.
        self.theta_obs_car = 0.
        # reward of V
        self.gparameter, self.Vxf = load_V()

        self.viewer = None
        # self.reset()

        self.reward_V = 0.
        self.reward_V_ = 0.
        self.reward_V_mid = 0.
        self.obs_avoid_reward = 0.
        self.delta_reward_V = 0.
        self.reward_stop = 0.
        self.reward_acc = 0.
        self.reward_spin = 0.
        self.bound_reward = 0.
        self.length_reward = 0.
        self.total_reward = 0.

        self.u = 0.
        self.length = 0  # the MAX length of episode

        self.dist_vaild = 0.
        self._dist_vaild = 0.
        self.obs_theta = 0.
        self.obs_theta_ = 0.
        self.obs_theta_mid = 0.
        self.delta_obs_theta = 0.
        self.obs_avoid_w = 0.
        self.error_theta = self._get_goal_theta_in_car() - self.sim.theta
        self.dist_k = 2.5

        self.stop_count = 0
        self.stop_count_total = 0

    def sample_action(self):
        if self.is_discrete_action:
            action = np.random.choice(list(range(3)))
        else:
            action = np.random.uniform(self.action_space, size=2)
        return action

    def _is_goal_reached(self, k=1):
        goal = self._get_goal_dist() < k * self.dist_threshold
        return goal

    def _is_done(self):
        i = self.sim.is_invalid()
        c = self.crash()
        # ss = self._spin_stop()
        g = self._is_goal_reached()
        s = self._stop()
        t = self.sim.time > 25
        # print('i',i)
        # print('c',c)
        # print('ss',ss)
        # print('g',g)
        # print('s',s)
        # print('t',t)
        return i or c or g or s or t

    def _get_dist(self, p1: np.ndarray, p2: np.ndarray):
        return np.linalg.norm(p1 - p2)

    def _get_theta(self):
        return np.abs(self.sim.theta) < 0.1

    def _get_all_obs_theta_in_car(self):
        obs_car_dist = self.obstacle - self.sim.position.reshape(2, -1)
        rotation_matrix = np.array([
            [np.cos(self.dx_car_theta), np.sin(self.dx_car_theta)],
            [-np.sin(self.dx_car_theta), np.cos(self.dx_car_theta)]
        ])
        position_obs_in_car = rotation_matrix.dot(obs_car_dist)
        theta_obs_car = np.arctan2(position_obs_in_car[1, :], position_obs_in_car[0, :])
        theta_obs_car_mark = np.where(np.abs(theta_obs_car) < np.pi / 2.1, theta_obs_car, 0.0)
        return theta_obs_car_mark


    def _get_obs_theta_in_car(self):
        obs_car_dist = self.obstacle[:, self.obs_index] - self.sim.position
        self.positon_obs_in_car = np.array([
            [np.cos(self.dx_car_theta), np.sin(self.dx_car_theta)],
            [-np.sin(self.dx_car_theta), np.cos(self.dx_car_theta)]
        ]).dot(obs_car_dist)

        self.theta_obs_car = np.arctan2(self.positon_obs_in_car[1],
                                        self.positon_obs_in_car[0])

        if np.abs(self.theta_obs_car) < np.pi / 2:
            theta_obs_car = self.theta_obs_car
        else:
            theta_obs_car = 0.

        return theta_obs_car


    def _get_u_in_car(self):
        self.u_incar = np.array([
            [np.cos(self.sim.theta), np.sin(self.sim.theta)],
            [-np.sin(self.sim.theta), np.cos(self.sim.theta)]
        ]).dot(self.u)
        return self.u_incar

    def _get_v_in_car(self):
        self.dx_car_incar = np.array([
            [np.cos(self.sim.theta), np.sin(self.sim.theta)],
            [-np.sin(self.sim.theta), np.cos(self.sim.theta)]
        ]).dot(self.dx_car)
        return self.dx_car_incar

    def _get_goal_theta_in_car(self):

        self.theta_goal_car = np.arctan2(self.goal_pos[1] - self.sim.position[1],
                                         self.goal_pos[0] - self.sim.position[0] + 1e-10)
        return self.theta_goal_car

    def _get_goal_dist(self):
        return self._get_dist(self.get_virtual_position(), self.goal_pos)


    def _stop(self):
        if self._get_dist(self.sim.position, self.sim._state[[2, 3]]) < 0.00001 and self._get_dist(self.sim.position,
                                                                                                   self.goal_pos) < 0.1:
            self.stop_count += 1
            self.stop_count_total += 1

        if self.stop_count > 10 and self._get_dist(self.sim.position,
                                                   self.sim._state[[2, 3]]) < 0.00001 and self._get_dist(
                self.sim.position, self.goal_pos) < 0.1:
            self.stop_count = 0
            print('V = 0!')
        else:
            self.reward_stop = 0
        return self.stop_count_total > 800

    def crash(self):
        c = self.barrier(self.get_virtual_position(), self.obstacle, k=1) > 1e-4
        if c:
            print('################## Crashed!')
            # print('################## Obstacle=', self.obs_pos)
        return c

    def euclidean_distance(self, x, x_obs, Sum=True):
        diff = x - x_obs
        squared_diff = np.square(diff)
        if Sum is True:
            summed = np.sum(squared_diff, axis=0)
        else:
            summed = squared_diff
        distance = np.sqrt(summed)
        return distance  # 1xN

    def barrier(self, x, x_so, obs_num=-1, k=1.):  # Returns the distance between the robot and the obstacle within the collision threshold
        g = [np.array(())]
        G = []
        G_new = []
        G_obs = []
        if self.random_obs == True:
            obs_r = self.r_rand
        else:
            obs_r = self.r

        if obs_num == -1:
            r = obs_r
        else:
            r = obs_r[obs_num]
        R = self.R
        gain = 1
        xi = np.array(1e-12)
        obs_R = np.array(np.add(r, R) * k)
        nbData = np.shape(x)
        nbObs = len(x_so.T)
        if nbObs <= 2:
            x_obs = x_so
            a = self._get_dist(x, x_obs)
            # a = np.reshape(a, [1, nbData])
            delta = a - obs_R
            c = np.sqrt(np.square(delta) + 4 * xi)
            g = 0.5 * (c - delta)
            G.append(g * gain)
        else:
            num_obs = len(x_so[1, :])
            for i in range(num_obs):
                x_obs = x_so[:, i]
                a = self._get_dist(x, x_obs)
                # a = np.reshape(a, [1, nbData])
                delta = a - obs_R[i]
                c = np.sqrt(np.square(delta) + 4 * xi)
                g = 0.5 * (c - delta)
                G.append(g * gain)
        G_obs = np.max(G)
        return G_obs

    def get_obs_dist(self, x, x_obs):
        if self.random_obs == True:
            obs_r = self.r_rand
        else:
            obs_r = self.r
        dist = self._get_dist(x, x_obs)
        obs_dist = dist - self.R - obs_r[self.obs_index]
        return obs_dist

    def obs_check(self, x, x_so, r, k=1.):  # Find out which obstacle is nearest and return the index
        R = self.R
        G = []
        xi = np.array(0.00000000001)
        # obs_R = np.array((r + R) * k)
        for i in range(len(x_so[1, :])):
            x_obs = x_so[:, i]
            a = self._get_dist(x, x_obs)

            delta = a - np.array((r[i] + R) * k)
            c = np.sqrt(np.square(delta) + 4 * xi)
            g = 0.5 * (c - delta)
            G.append(g.squeeze())
        obs_theta_mark = self._get_all_obs_theta_in_car()
        mark = np.abs(obs_theta_mark)>0
        G_mark = np.where(mark, G, 0)
        self.obs_index =np.argmax(G_mark)
        # self.obs_index = G.index(max(G))
        return self.obs_index

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _get_real_reward(self):
        # boundary
        if self.sim.is_invalid():
            self.bound_reward = 1e2 * self.reward_V_  # * self.reward_V
        else:
            self.bound_reward = 0

        if self.delta_obs_theta > 1e-4 and self.sim.v > 0:  # and np.abs(self.obs_theta_) > np.pi/6
            self.obs_avoid_reward = -0.95 * self.reward_V_
        else:
            self.obs_avoid_reward = 0.


        self.total_reward = self.reward_V_ + self.bound_reward + self.obs_avoid_reward  # 42
        self._print_info(self.total_reward)
        return self.total_reward

    # tiv reward
    def _get_tiv_reward(self):
        u, w = self.sim.speed
        x, y = self.sim.position
        theta = self.sim.theta

        reward_distance = 0

        v_bound_1 = 0.1
        v_bound_2 = 0.5

        if self.sim.v < v_bound_1:
            r_scr = np.log((self.sim.v) / v_bound_1) + 5
        elif self.sim.v < v_bound_2:
            r_scr = -15 * (self.sim.v - v_bound_1) / (v_bound_2 - v_bound_1)
        else:
            r_scr = -10

        # delta_a
        delta_a = np.abs(u - self.sim._state[0]) / 0.001
        r_delta_a = -(delta_a ** 2) / 3600

        # Check correctness
        if self._is_goal_reached():
            r_goal = 100 / self.sim.time
        else:
            r_goal = 0

        # Check correctness
        if self.sim.is_invalid() or self.crash is True:
            r_crash = -100
        else:
            r_crash = 0
        total_reward = r_scr + r_delta_a + r_goal + r_crash
        return total_reward

    # GUO reward
    def _get_defeat2_reward(self):
        far_value = 0.2
        far_value = 1e-4
        far = np.linalg.norm(self.sim.position - self.obstacle[:, self.obs_index]) > (self.obs_threshold + far_value)

        # near count
        near_value = 0.1
        near = self._get_dist(self.sim.position, self.obstacle[:, self.obs_index]) <= (self.obs_threshold + near_value)
        # obstacle Reward
        obstacle_cost = -100 if near else 0.1

        # Directional Reward
        reward_directional = (np.pi - np.abs(self.sim.theta) * 5) * 0.1
        if reward_directional < 0:
            reward_directional *= 4
            if reward_directional < -np.pi * 2:
                reward_directional = -np.pi * 2

        # Distance reward ---sparse
        last_goal_dis = np.linalg.norm(self._last_pos - self.goal_pos)
        goal_distance = np.linalg.norm(self.sim.position - self.goal_pos)
        tar_velocity = 0.03
        target_reward = np.max([-tar_velocity, np.min([last_goal_dis - goal_distance, tar_velocity])])
        self._last_pos = self.sim.position

        # weight
        reach_weight = 1000  # + 100/self.sim.time
        target_weight = 100
        obs_weight = 1
        far_weight = 0.2
        direction_weight = 0
        is_invalid_weight = 0.1

        # sum(reward)
        goal_reward = target_reward * target_weight + reach_weight * self._is_goal_reached() + 0.1 / self.sim.time
        final_reward = goal_reward + \
                       obs_weight * obstacle_cost + \
                       far * far_weight + \
                       reward_directional * direction_weight - \
                       is_invalid_weight * self.sim.is_invalid()

        return final_reward

    def _get_observation(self):
        x, y = self.get_virtual_position()
        theta = self.sim.theta
        v = self.sim.v
        phi = self.sim.phi

        if self.random_obs == True:
            r = self.r_rand
        else:
            r= self.r

        self.obs_index = self.obs_check(self.get_virtual_position(), self.obstacle, r,
                                        k=self.dist_k)
        self.dist_vaild = self.barrier(self.get_virtual_position(), self.obstacle[:, self.obs_index], self.obs_index,
                                       k=self.dist_k)
        # self._dist_vaild = self.barrier(self.sim._state[[2, 3]], self.obstacle, k=self.dist_k)


        self.obs_theta_ = self._get_obs_theta_in_car()
        # self.obs_theta_mid = np.copy(self.obs_theta_)


        self.delta_obs_theta = np.abs(self.obs_theta_) - np.abs(self.obs_theta)

        if self.dist_vaild > 1e-4:
            self.obs_theta_ = self.obs_theta_
            self.obs_pos = self.obstacle[:, self.obs_index]
            # obs_embeding = 0
        else:
            self.obs_theta_ = 1e-8
            self.obs_pos = self.obstacle[:, self.obs_index]

        option = 'rl'
        # option = 'GMR'
        self.dx_u, self.u, Vx, Vdot, self.V = ds_stabilizer(self.get_virtual_position(), self.dist_vaild,
                                                            self.obs_theta_, self.delta_obs_theta,
                                                            self.sim.theta, self.sim.phi,
                                                            option, self.Vxf,
                                                            self.gparameter, 0,
                                                            dx_rl=self.dx_car, x_obs=self.obstacle,
                                                            r=self.r)

        self.reward_V_ = -0.5 * self.V
        self.reward_V_mid = np.copy(self.reward_V_)
        self.delta_reward_V = self.reward_V_mid - self.reward_V

        self.Vx = Vx
        self.Vdot = Vdot


        # Orientation error
        theta_to_goal = self._get_goal_theta_in_car()
        if theta_to_goal < 0:
            theta_to_goal += 2 * np.pi
        else:
            theta_to_goal = theta_to_goal

        dir_to_goal = [self.goal_pos[1] - self.sim.position[1], self.goal_pos[0] - self.sim.position[0]]


        rad_error_theta = np.dot(100 * self.dx_car, dir_to_goal) / (
                    np.linalg.norm(100 * self.dx_car) * np.linalg.norm(dir_to_goal) + 1e-12)
        self.error_theta = np.arccos(rad_error_theta)
        print('##### goal_error_theta:', np.degrees(self.error_theta))

        self.pos_info = np.hstack([
            self.goal_dist,
            x, y,
            theta,
            v,
            phi,
            self.dist_vaild,
            self.obs_theta_,
            self.a_ref,
            self.w_ref
        ])

        return self.pos_info

    def normalization(self, pos_info):    # pass
        pos_info[0] = (pos_info[0] - self.goal_dist_low) / (self.goal_dist_high - self.goal_dist_low)
        pos_info[1] = (pos_info[1] - self.FIELD_SIZE_x_low) / (self.FIELD_SIZE_x_up - self.FIELD_SIZE_x_low)
        pos_info[2] = (pos_info[2] - self.FIELD_SIZE_y_low) / (self.FIELD_SIZE_y_up - self.FIELD_SIZE_y_low)
        pos_info[3] = (pos_info[3] - self.theta_low) / (self.theta_high - self.theta_low)
        pos_info[4] = (pos_info[4] - self.v_low) / (self.v_high - self.v_low)
        pos_info[-5] = (pos_info[-5] - self.phi_low) / (self.phi_high - self.phi_low)
        return pos_info

    def step(self, action):
        # self.render()
        a, w = action

        self.reward_V = self.reward_V_mid  # pass

        self.obs_theta = self.obs_theta_

        dx_rad = np.arctan2(self.dx_car[1], self.dx_car[0])
        dxu_rad = np.arctan2(self.dx_u[1], self.dx_u[0])
        if dx_rad < 0:
            dx_rad += 2 * np.pi
        if dxu_rad < 0:
            dxu_rad += 2 * np.pi

        theta_last = self.sim.theta  # sim.theta ~ [-pi, pi]
        if theta_last < 0:
            theta_last += 2 * np.pi  # theta_last ~ [0, 2pi]


        dxu_theta = max(theta_last, dxu_rad) - min(theta_last, dxu_rad)

        print('##### delta_dxu/theta_deg_MAXMIN:', np.degrees(dxu_theta))

        v_modify = np.linalg.norm(self.dx_u)
        a_ref = 0.3 * v_modify * np.cos(dxu_theta)  # v * cos

        if a_ref < 0:
            a_ref = 0 * a_ref
        else:
            a_ref = a_ref

        # obs_0_all = False
        obs_0_all = True


        du_theta_rad = dxu_rad - theta_last  # The relative difference between the dx and du angles after conversion to [0,2*pi]
        if du_theta_rad < 0:
            du_theta_rad += 2 * np.pi

        w_ref = du_theta_rad

        if 0 < du_theta_rad < np.pi / 2:
            if self.sim.v > 0:
                w_ref = du_theta_rad
            else:
                w_ref = -du_theta_rad
        elif np.pi / 2 < du_theta_rad < np.pi:
            if self.sim.v > 0:
                w_ref = du_theta_rad
            else:
                w_ref = -du_theta_rad
        elif np.pi < du_theta_rad < 3 * np.pi / 2:
            if self.sim.v > 0:
                w_ref = du_theta_rad - 2 * np.pi
            else:
                w_ref = 2 * np.pi - du_theta_rad
        elif 3 * np.pi / 2 < du_theta_rad < 2 * np.pi:
            if self.sim.v > 0:
                w_ref = du_theta_rad - 2 * np.pi
            else:
                w_ref = 2 * np.pi - du_theta_rad


        if np.linalg.norm(self.dx_u) < 0.0005:
            w_ref = 0
        else:
            w_ref = w_ref

        if obs_0_all is True:
            if self.dist_vaild > 1e-4 and np.abs(self.obs_theta_) > 1e-8:
                a_ref = 0.
                w_ref = 0.
            print('##### self.delta_dudx_rads:', self.delta_v_rad_last)

        else:
            a_ref = a_ref
            w_ref = w_ref

        self.a_ref = np.clip(a_ref, self.MIN_ACC, self.MAX_ACC)
        self.w_ref = np.clip(w_ref, -self.MAX_omiga, self.MAX_omiga)

        # boundary clip
        self.a_real = np.clip(a, self.MIN_ACC, self.MAX_ACC)
        self.w_real = np.clip(w, -self.MAX_omiga, self.MAX_omiga)

        self.sim.step(np.array([self.a_real, self.w_real]))
        self.num_step += 1

        dx_car = self.sim.position - self.sim._state[[2, 3]]
        self.dx_car_theta = np.arctan2(dx_car[1], dx_car[0])

        if self.dx_car.all() == 0:
            self.dx_car = np.array([0.001 * np.cos(self.sim.theta), 0.001 * np.sin(self.sim.theta)])
        else:
            self.dx_car = dx_car

        # compute distance rate
        self.goal_dist = self._get_goal_dist()
        dist_rate = self.goal_dist / self.orig_dis

        # crash check
        c = self.crash()

        return (self._get_observation()), (self._get_real_reward()), self._is_done(), self._is_done(), {'d': dist_rate,
                                                                                                        'c': c
            , 'action_ref': np.array([self.a_ref, self.w_ref])
            , 'action_real': np.array([self.a_real, self.w_real])
            , 'num_step': self.num_step}

    def _print_info(self, reward):
        frequency = 100
        if self._is_done() or self.sim.ticks % np.round(1 / self.dt / frequency) == 0:
            # u, w = self.sim.speed
            a = self.sim.a
            w = self.sim.w
            v = self.sim.v
            theta = self.sim.theta
            phi = self.sim.phi
            x, y = self.sim.position
            dist = self._get_dist(self.sim.position, self.goal_pos)
            rad_dx = np.arctan2(self.dx_car[1], (self.dx_car[0] + 1e-20))  # next step，dx_car is the current dx
            degree_dx = np.degrees(rad_dx)
            rad_dx_u = np.arctan2(self.dx_u[1], (self.dx_u[0] + 1e-20))  # next step，dx_u is the current du
            degree_dx_u = np.degrees(rad_dx_u)

        print(f"T {self.sim.time}_"
              f"Pos({x:.4f}, {y:.4f}), "
              f"Theta({theta:.4f}), "
              f"phi({phi:.4f}), "
              f"v {v:.4f}, "
              # f"a({a:.4f}, {w:.4f}), "
              f"a_real({self.a_real:.4f}, {self.w_real:.4f}), "
              f"a_ref({self.a_ref:.4f}, {self.w_ref:.4f}), "
              f"r_v_ {self.reward_V_:.4f},\n "
              f"R {self.total_reward:.4f}, "
              f"dist({dist:.4f}), "
              f"Obs_d({self.pos_info[-4]:.4f}), "
              f"Obs_theta({self.pos_info[-3]:.4f}), "
              f"dx({degree_dx:.4f}), "
              f"dx_u({degree_dx_u:.4f}),"
              f"Vx([{self.Vx}],"
              f"Vdot([{self.Vdot}])"
              )

    def reset(self, pos_ini=None, seed: Optional[int] = 2333, options: Optional[dict] = None):
        super().reset(seed=seed)

        # reset simulation
        self.stop_count = 0
        self.stop_count_total = 0
        self.u = 0
        self.reset_count = 0
        continue_reset = True

        self.goal_pos = np.array([0, 0])  # The actual location of the target point, the original position coordinate (0,0)
        self.sim.reset(self.ep, intial_diy=False, fixed=False, pos_ini=pos_ini)  # Reset robot position
        self.ep += 1
        self.orig_dis = self._get_dist(self.sim.goal_pos, self.sim.start_point)
        if self.random_obs == True:

            self.obstacle_random[:, -1] = (self.sim.start_point - self.goal_pos) / 2
            for i in range(len(self.r_rand)-1):
                valid_position = False
                attempts = 0
                max_attempts = 3000

                while not valid_position and attempts < max_attempts:
                    attempts += 1

                    x = np.random.uniform(self.FIELD_SIZE_x_low, self.FIELD_SIZE_x_up)
                    y = np.random.uniform(self.FIELD_SIZE_y_low, self.FIELD_SIZE_y_up)

                    if np.isclose(x, 0.0) and np.isclose(y, 0.0):
                        continue

                    obs_to_goal = self._get_dist([x, y], self.goal_pos)
                    obs_to_start = self._get_dist([x, y], self.sim.start_point)

                    conflict = False
                    for k in range(i):
                        prev_obs = self.obstacle_random[:, k]
                        if self._get_dist([x, y], prev_obs) <= 2.3:
                            conflict = True
                            break

                    if (8.0> obs_to_goal > 2.) and \
                            (8.0> obs_to_start > 2.0) and \
                            (not conflict):

                        if not (np.isclose(x, self.goal_pos[0]) and np.isclose(y, self.goal_pos[1])):
                            self.obstacle_random[:, i] = [x, y]
                            valid_position = True

                if attempts >= max_attempts:
                    raise RuntimeError(f"Obstacle{i}Failed to generate, the number of attempts exceeded{max_attempts}")

            self.obstacle = self.obstacle_random
        else:
            self.obstacle = self.obstacle_origin
        print("obstacle", self.obstacle)

        self.pos_info = np.hstack([
            self._get_goal_dist(),
            self.sim.position[0], self.sim.position[1],
            self.sim.theta,
            self.sim.v,
            self.sim.phi,
            self.dist_vaild,
            self.obs_theta_,
            self.a_ref, self.w_ref
        ])
        self._print_info(0)
        info = {}
        return (self._get_observation()), info

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer(self)
        return self.viewer.render(mode)

    def close(self):
        if self.viewer:
            self.viewer.viewer.close()
            self.viewer = None



if __name__ == '__main__':
    env = CarEnv()

