import numpy as np
import pygame
import new_env_parameters_setting_sb3 as env_para


class Viewer:
    def __init__(self, env):
        self.env_para_ = env_para.parameters()
        self.env = env
        self.sim = env.sim
        self.pathTrace = 100  # Total number of trace points
        self.pathTraceSpace = 1  # Track update interval

        self.pathTraceSpaceCounter = 0
        self.path = np.zeros([self.pathTrace, 2])
        self.pathPtr = 0
        self.r = self.env_para_.r

        field_size_x_low = self.sim.FIELD_SIZE_x_low
        field_size_x_up = self.sim.FIELD_SIZE_x_up
        field_size_y_low = self.sim.FIELD_SIZE_y_low
        field_size_y_up = self.sim.FIELD_SIZE_y_up

        field_width = field_size_x_up - field_size_x_low
        field_height = field_size_y_up - field_size_y_low

        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.scale = min(self.screen_width / field_width,
                         self.screen_height / field_height)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robot Simulation")

        self.green = (0, 255, 0)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)

        self.trail_colors = {
            'start': (255, 165, 0),
            'end': (0, 191, 255)
        }
        self.trail_sizes = {
            'min': 2,
            'max': 5
        }


        self.trail_params = {
            'max_frames': 3000,
            'frame_interval': 18,
            'base_alpha': 100,
            # 'alpha_decay': 0.025,
            'alpha_decay': 0.1,
            'min_distance': 0.1,
            'size_scale': 1
        }
        self.trail_frames = []
        self.frame_counter = 0

        self.trail_points = []

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height),
                                              pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("Smooth Robot Simulation")

        random = False

        if random == True:
            self.r = self.env_para_.r_rand
        else:
            self.r = self.env_para_.r

        self.obstacleobj = []
        self.obstacle_t = []
        num_obs = int(len(self.env.obstacle[0]))
        for i in range(num_obs):
            obs_radius = self.r[i]
            self.obstacleobj.append(
                pygame.draw.circle(self.screen, self.black, (0, 0), obs_radius)
            )

        self.robot_img = pygame.image.load("assets/car2.png")
        self.robot_img = pygame.transform.scale(
            self.robot_img,
            (int(self.robot_img.get_width() * 0.1),
             int(self.robot_img.get_height() * 0.1))
        )
        self.robot_rect = self.robot_img.get_rect(center=(0, 0))

        self.car_trail_images = []

        self.car_image = pygame.image.load("assets/car2.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image,
                                                (int(self.car_image.get_width() * 0.1),
                                                 int(self.car_image.get_height() * 0.1)))

    def reset_trail(self):
        self.trail_frames.clear()
        self.trail_points.clear()
        self.car_trail_images.clear()

        self.path.fill(0)
        self.pathPtr = 0

        self.frame_counter = 0
        self.pathTraceSpaceCounter = 0

    def update_trail(self):
        self.frame_counter += 1

        if self.frame_counter % self.trail_params['frame_interval'] == 0:
            current_pos = self.sim.position.copy()

            if len(self.trail_frames) > 0:
                last_pos = self.trail_frames[-1]['position']
                if np.linalg.norm(current_pos - last_pos) < self.trail_params['min_distance']:
                    return

            self.trail_frames.append({
                'position': current_pos,
                'angle': self.sim.theta,
                'alpha': self.trail_params['base_alpha'],
                'scale': 1.0
            })

        for frame in self.trail_frames:
            frame['alpha'] = max(0, frame['alpha'] - self.trail_params['alpha_decay'])
            frame['scale'] *= self.trail_params['size_scale']

        self.trail_frames = [f for f in self.trail_frames if f['alpha'] > 5]

    def draw_trail(self):
        for frame in self.trail_frames:
            x = (frame['position'][0] - self.sim.FIELD_SIZE_x_low) * self.scale
            y = self.screen_height - (frame['position'][1] - self.sim.FIELD_SIZE_y_low) * self.scale

            scaled_image = pygame.transform.scale(
                self.car_image,
                (int(self.car_image.get_width() * frame['scale']),
                 int(self.car_image.get_height() * frame['scale']))
            )
            alpha_surface = pygame.Surface(scaled_image.get_size(), pygame.SRCALPHA)
            alpha_surface.fill((255, 255, 255, frame['alpha']))
            scaled_image.blit(alpha_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            rotated_image = pygame.transform.rotate(scaled_image, np.degrees(frame['angle']) - 90)
            rect = rotated_image.get_rect(center=(x, y))
            self.screen.blit(rotated_image, rect)

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reset_trail()

        self.screen.fill((255, 255, 255))

        num_obs = int(len(self.env.obstacle[0]))
        for i in range(num_obs):
            obs_x = (self.env.obstacle[0][i] - self.sim.FIELD_SIZE_x_low) * self.scale
            obs_y = self.screen_height - (self.env.obstacle[1][i] - self.sim.FIELD_SIZE_y_low) * self.scale
            pygame.draw.circle(self.screen, self.black,
                               (int(obs_x), int(obs_y)),
                               int(self.r[i] * self.scale))

        to_remove = []
        for idx, trail in enumerate(self.car_trail_images):
            screen_x = (trail['pos'][0] - self.sim.FIELD_SIZE_x_low) * self.scale
            screen_y = self.screen_height - (trail['pos'][1] - self.sim.FIELD_SIZE_y_low) * self.scale

            car_surface = pygame.Surface(self.car_image.get_size(), pygame.SRCALPHA)
            car_surface.blit(self.car_image, (0, 0))
            car_surface.set_alpha(trail['alpha'])

            rotated_surface = pygame.transform.rotate(car_surface, np.degrees(trail['angle']) - 90)
            rect = rotated_surface.get_rect(center=(screen_x, screen_y))

            self.screen.blit(rotated_surface, rect.topleft)
            self.car_trail_images[idx]['alpha'] -= self.car_trail_params['alpha_decay']

            if self.car_trail_images[idx]['alpha'] <= 0:
                to_remove.append(idx)

        for idx in reversed(to_remove):
            self.car_trail_images.pop(idx)

        robot_screen_x = (self.sim.position[0] - self.sim.FIELD_SIZE_x_low) * self.scale
        robot_screen_y = self.screen_height - (self.sim.position[1] - self.sim.FIELD_SIZE_y_low) * self.scale
        rotated_img = pygame.transform.rotate(self.car_image, np.degrees(self.sim.theta) - 90)
        self.screen.blit(rotated_img, rotated_img.get_rect(center=(robot_screen_x, robot_screen_y)))

        self.update_trail()
        self.draw_trail()

        self.pathTraceSpaceCounter = (self.pathTraceSpaceCounter + 1) % self.pathTraceSpace
        if self.pathTraceSpaceCounter == 0:
            self.path[self.pathPtr] = self.sim.position.copy()
            self.pathPtr = (self.pathPtr + 1) % self.pathTrace

        for i in range(self.pathTrace):
            idx = (self.pathPtr + i) % self.pathTrace
            age_ratio = i / (self.pathTrace - 1)

            radius = 1 + 4 * age_ratio  # 2px到8px渐变

            color = (
                int(0 * (1 - age_ratio) + 255 * age_ratio),  # R
                int(100 * (1 - age_ratio) + 165 * age_ratio),  # G
                int(200 * (1 - age_ratio) + 0 * age_ratio)  # B
            )

            point = self.path[idx]
            screen_x = (point[0] - self.sim.FIELD_SIZE_x_low) * self.scale
            screen_y = self.screen_height - (point[1] - self.sim.FIELD_SIZE_y_low) * self.scale

            trail_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, color, (radius, radius), int(radius))
            self.screen.blit(trail_surface, (screen_x - radius, screen_y - radius))

        robot_x = (self.sim.position[0] - self.sim.FIELD_SIZE_x_low) * self.scale
        robot_y = self.screen_height - (self.sim.position[1] - self.sim.FIELD_SIZE_y_low) * self.scale
        rotated = pygame.transform.rotate(self.robot_img, np.degrees(self.sim.theta) - 90)
        self.screen.blit(rotated, rotated.get_rect(center=(robot_x, robot_y)))


        goal_screen_x = (self.env.goal_pos[0] - self.sim.FIELD_SIZE_x_low) * self.scale
        goal_screen_y = self.screen_height - (self.env.goal_pos[1] - self.sim.FIELD_SIZE_y_low) * self.scale
        pygame.draw.circle(self.screen, self.green,
                           (int(goal_screen_x), int(goal_screen_y)),
                           int(0.3 * self.scale))

        pygame.display.flip()
        return None if mode == 'human' else pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()