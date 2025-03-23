import numpy as np


class parameters:
    def __init__(self):
        # Time delta per step
        self.dt = 0.01
        # Environment parameters
        # self.seed(2)
        self.testItr = 200.

        # Boundaries of the action
        self.MIN_ACC = -0.5
        self.MAX_ACC = 1.5
        self.MAX_omiga = np.pi / 2
        self.is_discrete_action = False
        self.stop = 0.0

        # how close to goal = reach goal
        self.dist_threshold = 0.2

        # how close to obstacle = crash obstacle
        self.obs_threshold = 1
        # self.obs_threshold = 0.4
        self.obs_index = 0.

        # Action and observation spaces
        self.FIELD_SIZE_x_low = -3.                    # SMALL
        self.FIELD_SIZE_x_up = 8.
        self.FIELD_SIZE_y_low = -3.
        self.FIELD_SIZE_y_up = 8.
        self.u_uobound = 1.
        self.u_lowbound = 0.

        # self.FIELD_SIZE_x_low = -5.                  # BIG
        # self.FIELD_SIZE_x_up = 40.
        # self.FIELD_SIZE_y_low = -10.
        # self.FIELD_SIZE_y_up = 35.


        # self.obstacle = np.array([[-7, 10, -17, 17, 15, -22, -8],
        #                   [-13, 6, 0, -13, 0, -6, 13]])

        self.obstacle_no = np.array([[80., 80., 80., 80., 80., 80., 80.],
                                     [80., 80., 80., 80., 80., 80., 80.]])
        self.obstacle = np.array([[2, 2.5, 3.5, 1.5, 4., 6., 5.],  # , 5.7 , 5.5SMALL
                                  [1.1, 3.5, -0.8, 6., 7., 3, 2.]])
        # self.obstacle = np.array([[80., 80., 80., 80., 80., 80., 80.],
        #                              [80., 80., 80., 80., 80., 80., 80.]])

        # self.r = [0.4, 0.4, 0.45, 0.4, 0.35, 0.4, 0.4]
        self.r = [0.42, 0.4, 0.42, 0.4, 0.43, 0.39, 0.4]
        self.r_rand = [0.42, 0.4, 0.42, 0.4, 0.43, 0.39, 0.4]
        self.obs_num = len(self.obstacle[0, :])


        self.R = 0.18
        self.L = 0.29
        self.goal_pos = np.array([0, 0])


        self.theta_low = -np.pi
        self.theta_high = np.pi
        # self.v_low = -0.5
        self.v_low = -0.4
        # self.v_high = 0.8
        self.v_high = 0.6

        # self.phi_low = -np.pi / 10
        # self.phi_high = np.pi / 10
        self.phi_low = -np.pi / 8
        self.phi_high = np.pi / 8
        self.error_theta_low = -3/2*np.pi
        self.error_theta_high = 3/2*np.pi
        # self.obs_dist_low = 0.
        # self.obs_dist_high = 1.5  # 达到碰撞检测阈值时的agv与最近障碍物的距离
        self.obs_theta_low = -np.pi
        self.obs_theta_high = np.pi
        self.goal_dist_low = 0.
        self.goal_dist_high = 10.
        self.ux_low = -5.
        self.ux_high = 5.
        self.uy_low = -5.
        self.uy_high = 5.
        self.delta_v_rad_low = -np.pi/2
        self.delta_v_rad_high = np.pi/2
