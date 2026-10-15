import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np

import torch
from numpy.linalg import norm
from utils_trans.human import Human
from utils_trans.robot import Robot
from utils_trans.state import ObservableState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib_opt import *

class CrowdSim:

    n_laser = 1800
    laser_angle_resolute = 0.003490659
    laser_min_range = 0.27
    laser_max_range = 6.0
    square_width = 10.0

    def __init__(self, args, terrain_dim, action_range, device, action_choices=None, quadruped_policy=None, quadruped_env=None):
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.terrain_dim = terrain_dim
        self.robot_policy = args.policy
        self.robot_model = args.robot_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        self.frame_stack = args.frame_stack
        self.lidar_obs_type = args.lidar_obs_type
        self.random_human_v_pref = args.random_human_v_pref
        self.device = device
        self.action_penalty = args.action_penalty
        self.penalty_rotation = args.penalty_rotation
        self.clip_action = args.clip_action
        # last-time distance from the robot to the goal
        self.goal_distance_last = None
        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32) # used for visualization

        # laser state
        self.scan_current = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        self.scan_end_current = np.zeros((self.n_laser, 2), dtype=np.float32)

        self.scan_last = np.zeros((self.frame_stack - 1, self.n_laser), dtype=np.float32)
        self.scan_end_last = np.zeros((self.frame_stack - 1, self.n_laser, 2), dtype=np.float32)
        self.lidar_obs = np.zeros((1, self.frame_stack, self.n_laser), dtype=np.float32)
        
        self.global_time = 0.0
        self.global_step = 0
        self.time_limit = 50
        self.time_step = 0.2
        self.y_range = 7.0
        self.x_range = 7.0
        self.v_min = 0.1
        self.max_episode_step = int(self.time_limit / self.time_step)
        self.randomize_attributes = False
        self.success_reward = 0.5
        self.collision_penalty = -0.5
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = 0.4
        self.goal_distance_factor = 1.4
        self.linear_penalty = -0.05
        self.nonstuck_threshold = 0.2  # m/s

        # here, more lines can be added to simulate obstacles
        # here, more lines can be added to simulate obstacles
        self.lines = np.zeros((4, 2, 2), dtype=np.float32)
        margin = [10.0, 10.0]
        self.lines[0, :, :] = np.array([[-margin[0], -margin[1]],
                                        [-margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[1, :, :] = np.array([[-margin[0],  margin[1]],
                                        [margin[0],  margin[1]]], dtype=np.float32) 
        self.lines[2, :, :] = np.array([[margin[0],  margin[1]],
                                        [margin[0], -margin[1]]], dtype=np.float32) 
        self.lines[3, :, :] = np.array([[margin[0], -margin[1]],
                                        [-margin[0], -margin[1]]], dtype=np.float32) 
        self.circle_radius = 4.0 # human distribution margin
        self.static_obstacle_area_x = 3.0 # static obstacle distribution area
        self.static_obstacle_area_y = 1.5 
        self.human_num_max = 4
        self.static_obstacle_num_max = 3
        self.robot_visible_threshold = 1.5
        self.rectangle_x = 4.0
        self.rectangle_y = [2.0, 5.0]
        self.add_human_num_max = 3
        self.add_human_num = None

        self.human_num = None
        self.static_obstacle_num = None

        self.humans = None
        self.human_v_pref = (0.5, 1.5)
        self.static_obstacles = None
        self.rectangles = None
        self.action_range = action_range
        self.action_choices = action_choices
        self.robot = Robot()
        self.robot.time_step = self.time_step
        self.robot.v_pref = action_range[1, 0]
        self.action_last = np.zeros(2) 

        self.position_noise = 0.03

        # quadruped robots
        self.quadruped_policy = quadruped_policy
        self.quadruped_env = quadruped_env
        if quadruped_env is not None:
            self.repeat_times = int(self.time_step / quadruped_env.unwrapped.step_dt)
            self.quadruped_num_env = quadruped_env.num_envs
        else:
            self.repeat_times = 0
            self.quadruped_num_env = 1
        
        self.specific_quadruped_envs_idx = 0
        self.init_quadruped_pose_w = None
 
        # visualization on 2D plane
        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # log lidar, robot, and humans
        self.log_env = {}

    def generate_random_static_obstacle(self):
        self.static_obstacle_num = int(np.random.randint(self.static_obstacle_num_max, size=1)[0] + 1)
        self.static_obstacles = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
        while True:
            positions_x = np.random.uniform(-self.static_obstacle_area_x, self.static_obstacle_area_x, 
                                            (self.static_obstacle_num, 1))
            positions_y = np.random.uniform(-self.static_obstacle_area_y, self.static_obstacle_area_y, 
                                            (self.static_obstacle_num, 1))
            radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            collision = False
            for i in range(self.static_obstacle_num):
                temp = False
                for j in range(i + 1, self.static_obstacle_num):
                    # allow 0.1m overlap
                    if hypot(positions_x[i, 0] - positions_x[j, 0], \
                             positions_y[i, 0] - positions_y[j, 0]) <= radiuses[i, 0] + radiuses[j, 0] - 0.1:
                        collision = True
                        temp = True
                        break
                if temp:
                    break
            if not collision:
                self.static_obstacles = np.hstack((positions_x, positions_y, radiuses))
                break

        
    def generate_random_human_position(self):
        self.human_num = int(np.random.randint(self.human_num_max, size=1)[0] + 1)
        self.humans = [None] * self.human_num
        for i in range(self.human_num):
            self.humans[i] = self.generate_circle_crossing_human()
        
        # add_human_num_up = int(np.random.randint(self.add_human_num_max + 1, size=1)[0])
        # for i in range(add_human_num_up):
        #     self.humans.append(self.generate_rectangle_flow_human(down=False))

        # add_human_num_down = int(np.random.randint(self.add_human_num_max + 1, size=1)[0])
        # for i in range(add_human_num_down):
        #     self.humans.append(self.generate_rectangle_flow_human(down=True))
        # self.add_human_num = add_human_num_up + add_human_num_down
        # self.human_num = self.human_num + self.add_human_num

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            human_policy.max_speed = self.humans[i].v_pref
            human_policy.radius = self.humans[i].radius
            human_policy.max_robot_speed = self.robot.v_pref
            self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        if self.static_obstacles is None:
            raise NotImplementedError(self.static_obstacles)
        human = Human()
        human.time_step = self.time_step

        human.radius = 0.3
        if self.random_human_v_pref:
            human.v_pref = np.random.uniform(*self.human_v_pref, 1)[0]
        else:
            human.v_pref = 1.0
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                if agent is None:
                    continue
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            for static_obs in range(self.static_obstacle_num):
                min_dist = human.radius + self.static_obstacles[static_obs, 2] + self.discomfort_dist
                if norm((px - self.static_obstacles[static_obs, 0], 
                         py - self.static_obstacles[static_obs, 1])) < min_dist:
                    collide = True
                    break
            if not collide:
                # px, py, gx, gy, vx, vy, theta
                human_theta = atan2(-py - py, -px - px)
                human.set(px, py, -px, -py, 0, 0, human_theta)
                break
        return human
    
    def generate_rectangle_flow_human(self, down=True):
        human = Human()
        human.time_step = self.time_step

        human.radius = 0.3
        if self.random_human_v_pref:
            human.v_pref = np.random.uniform(*self.human_v_pref, 1)[0]
        else:
            human.v_pref = 1.0
        while True:
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            px_side = np.random.random() > 0.5
            px = self.rectangle_x * np.power(-1, int(px_side)) + px_noise
            py = np.random.uniform(self.rectangle_y[0], self.rectangle_y[1], size=1)[0] * np.power(-1, int(down))
            gx = -px
            gy = py
            collide = False
            for agent in [self.robot] + self.humans:
                if agent is None:
                    continue
                min_dist = human.radius + agent.radius + 0.05
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            for static_obs in range(self.static_obstacle_num):
                min_dist = human.radius + self.static_obstacles[static_obs, 2] + 0.05
                if norm((px - self.static_obstacles[static_obs, 0], 
                         py - self.static_obstacles[static_obs, 1])) < min_dist:
                    collide = True
                    break
            if not collide:
                # px, py, gx, gy, vx, vy, theta
                human_theta = atan2(-py - py, -px - px)
                human.set(px, py, gx, gy, 0, 0, human_theta)
                break
        return human
        
    '''use c library to generate lidar scan, which is fast'''
    def get_lidar(self, isReset=False):
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        robot_pose_new = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        human_circles = -100.0 * np.ones((self.human_num_max + self.static_obstacle_num_max + self.add_human_num_max * 2, 3), dtype=np.float32)
        for i in range(num_circle_human):
            human_circles[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius], dtype=np.float32)
        for j in range(num_circle_obstacle):
            human_circles[j + num_circle_human] = np.array([self.static_obstacles[j, 0],
                                                            self.static_obstacles[j, 1],
                                                            self.static_obstacles[j, 2]], dtype=np.float32)
    
        scans_raw, scan_lines = cal_laser(human_circles, robot_pose_new)

        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32)
        scan_end = np.zeros((self.n_laser, 2), dtype=np.float32)
        for i in range(self.n_laser):
            ### used for visualization
            self.scan_intersection[i, 0, 0] = scan_lines[i, 0]
            self.scan_intersection[i, 0, 1] = scan_lines[i, 1]
            self.scan_intersection[i, 1, 0] = scan_lines[i, 2]
            self.scan_intersection[i, 1, 1] = scan_lines[i, 3]
            scan_end[i, :] = np.array([self.scan_intersection[i, 1, 0], self.scan_intersection[i, 1, 1]])
            ### used for visualization
        
        if self.lidar_obs_type == 'transform':
            self.scan_current = np.clip(scans_raw, self.laser_min_range, self.laser_max_range)
            if isReset:
                self.scan_end_current = np.copy(scan_end)
                for frame_idx in range(self.frame_stack - 1):
                    self.scan_last[frame_idx] = np.copy(self.scan_current)
                    self.scan_end_last[frame_idx] = np.copy(self.scan_end_current)
            else:
                # t1 = time()
                scan_last_raw = transform_scan_last(self.scan_end_last, robot_pose_new)   
                # print('time: ', time() - t1)
                self.scan_last = np.clip(scan_last_raw, self.laser_min_range, self.laser_max_range)      
                for frame_idx in range(self.frame_stack - 2):
                    self.scan_end_last[self.frame_stack - frame_idx - 2] = np.copy(self.scan_end_last[self.frame_stack - frame_idx - 3])
                self.scan_end_current = np.copy(scan_end)
                self.scan_end_last[0] = np.copy(self.scan_end_current)
        else:
            if isReset:
                self.scan_current = np.clip(scans_raw, self.laser_min_range, self.laser_max_range)
                for frame_idx in range(self.frame_stack - 1):
                    self.scan_last[frame_idx] = np.copy(self.scan_current)
            else:
                for frame_idx in range(self.frame_stack - 2):
                    self.scan_last[self.frame_stack - frame_idx - 2] = np.copy(self.scan_last[self.frame_stack - frame_idx - 3])
                self.scan_last[0] = np.copy(self.scan_current) 
                self.scan_current = np.clip(scans_raw, self.laser_min_range, self.laser_max_range) 

        lidar_obs = np.hstack((self.scan_current, self.scan_last.flatten()))
        self.lidar_obs = np.reshape(lidar_obs, (1, self.frame_stack, self.n_laser))

    def quadruped_w_2_init(self, robot_x_w, robot_y_w, robot_theta_w):
        dx = robot_x_w - self.init_quadruped_pose_w[0]
        dy = robot_y_w - self.init_quadruped_pose_w[1]
        c_theta = cos(self.init_quadruped_pose_w[2])
        s_theta = sin(self.init_quadruped_pose_w[2])
        x = dy * s_theta + dx * c_theta
        y = dy * c_theta - dx * s_theta
        theta = robot_theta_w - self.init_quadruped_pose_w[2]
        return x, y, theta

    def cal_dwa_action(self):
        # dwa calculate action
        action_cost = 99999.9
        action_dwa = np.zeros(2)
        dwa_horizon = 5
        for i in range(self.action_choices.shape[0]):
            robot_vel = self.action_choices[i]
            
            robot_x = self.robot.px
            robot_y = self.robot.py
            robot_theta = self.robot.theta
            collision = False
            dis_human_and_obstacle = 99999.9
            robot_theta = robot_theta + robot_vel[1] * self.time_step
            if robot_theta > np.pi:
                robot_theta -= (2.0 * np.pi)
            elif robot_theta < -np.pi:
                robot_theta += (2.0 * np.pi)
            for j in range(dwa_horizon):
                # differential model
                robot_x = robot_x + robot_vel[0] * self.time_step * cos(robot_theta)
                robot_y = robot_y + robot_vel[0] * self.time_step * sin(robot_theta)
                # differential model

                # distance to humans
                for k in range(self.human_num):
                    human_x = self.humans[k].px + (j + 1) * self.humans[k].vx * self.time_step
                    human_y = self.humans[k].py + (j + 1) * self.humans[k].vy * self.time_step
                    dis_human_temp = hypot(human_x - robot_x, human_y - robot_y)
                    if dis_human_temp <= self.humans[k].radius + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_human_temp)

                if collision:
                    break

                # distance to obstacles
                for k in range(self.static_obstacle_num):
                    obstacle_x = self.static_obstacles[k, 0]
                    obstacle_y = self.static_obstacles[k, 1]
                    dis_obstacle_temp = hypot(obstacle_x - robot_x, obstacle_y - robot_y)
                    if dis_obstacle_temp <= self.static_obstacles[k, 2] + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_obstacle_temp)

                if collision:
                    break

            if collision:
                continue
            dis_goal = hypot(self.robot.gx - robot_x, self.robot.gy - robot_y)
         
            action_cost_temp = 1.0 / (dis_human_and_obstacle + 0.8) + dis_goal * 0.2
            if action_cost > action_cost_temp:
                action_cost = action_cost_temp
                action_dwa = robot_vel
        return action_dwa

    def step(self, action, eval=False, save_data=False):
        human_actions = np.zeros((self.human_num, 2), dtype=np.float32)
        for i in range(self.human_num):
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != self.humans[i]]
            for k in range(self.static_obstacle_num):
                ob.append(ObservableState(
                        self.static_obstacles[k, 0], 
                        self.static_obstacles[k, 1], 
                        0.0, 0.0, self.static_obstacles[k, 2])
                        )
            if self.robot_visible_threshold * hypot(self.robot.vx, self.robot.vy) < hypot(self.humans[i].vx, self.humans[i].vy):
                ob.append(ObservableState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius))
                action_temp = self.humans[i].act(ob, has_robot=True)
                human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)
            else:
                action_temp = self.humans[i].act(ob)
                human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)

        done_quadruped = False
        if self.robot_model == 'quadruped':
            quadruped_obs = self.quadruped_env.get_observations()
            obs, history_obs = quadruped_obs['obs'].detach(), quadruped_obs['history_obs'].detach()
            obs_clone = obs.clone()
            action_after_clip = [action[0], action[1]]

            # clip action
            # clip_in_place_rotation
            if self.clip_action:
                if action[0] >= 0.3:
                    commands = torch.tensor([action[0], 0.0, action[1]], device=self.device)
                else:
                    if action[1] > 0.5:
                        action_after_clip = [0.0, 0.5]
                    elif action[1] < -0.5:
                        action_after_clip = [0.0, -0.5]
                    elif action[1] < 0.3 and action[1] > -0.3:
                        action_after_clip = [0.0, 0.0]
                    else:
                        action_after_clip = [0.0, action[1]]
                    commands = torch.tensor([0.0, 0.0, action_after_clip[1]], device=self.device)
            else:
                commands = torch.tensor([action[0], 0.0, action[1]], device=self.device)    

            # backward or forward clip
            # if self.clip_action:
            #     if action[0] >= 0.3 or action[0] <= -0.3:
            #         commands = torch.tensor([action[0], 0.0, action[1]], device=self.device)
            #     else:
            #         action_after_clip = [0.0, 0.0]
            #         commands = torch.tensor([0.0, 0.0, action_after_clip[1]], device=self.device)
            # else:
            #     commands = torch.tensor([action[0], 0.0, action[1]], device=self.device)   

            obs_clone[:, 6:9] = commands
            obs = obs_clone

            history_obs_clone = history_obs.clone()
            idx1 = 9 * 45 + 6
            idx2 = 9 * 45 + 9
            history_obs_clone[:, idx1:idx2] = commands
            history_obs = history_obs_clone

            for quadruped_step in range(self.repeat_times):
                with torch.inference_mode():
                    quadruped_action, encoder = self.quadruped_policy(obs, history_obs)
                    obs_dict, _, done_quadruped_tmp, _ = self.quadruped_env.step(quadruped_action, commands=[action[0], 0.0, action[1]])
                    obs, history_obs = obs_dict['obs'].detach(), obs_dict['history_obs'].detach()
                    encoder = encoder.detach()
                    if done_quadruped_tmp[self.specific_quadruped_envs_idx]:
                        done_quadruped = True
                        break
            robot_x_w, robot_y_w, robot_theta_w = self.quadruped_env.get_robot_pose(self.specific_quadruped_envs_idx)
            robot_x, robot_y, robot_theta = self.quadruped_w_2_init(robot_x_w, robot_y_w, robot_theta_w)
            if self.terrain_dim == 64:
                obs_clone = obs[self.specific_quadruped_envs_idx].cpu().numpy()
                encoder_clone = encoder[self.specific_quadruped_envs_idx].cpu().numpy()
                terrain = np.concatenate((obs_clone, encoder_clone)).flatten()
            else:
                terrain = None
        else:
            robot_theta = self.robot.theta + action[1] * self.time_step
            translation = action[0] * self.time_step
            robot_x = self.robot.px + translation * cos(robot_theta)
            robot_y = self.robot.py + translation * sin(robot_theta)
            if self.terrain_dim > 0:
                terrain = np.zeros(self.terrain_dim, dtype=np.float32)
            else: 
                terrain = None

        if robot_theta > np.pi:
            robot_theta -= (2.0 * np.pi)
        elif robot_theta < -np.pi:
            robot_theta += (2.0 * np.pi)
        
        actual_linear_v = hypot(robot_x - self.robot.px, robot_y - self.robot.py) / self.time_step
        actual_angular_v = (robot_theta - self.robot.theta) / self.time_step
        # update states
        self.robot.update_states(robot_x, robot_y, robot_theta, [actual_linear_v, actual_angular_v])
        
        for i in range(self.human_num):
            self.humans[i].update_states(human_actions[i])

        # get new laser scan and grid map
        self.get_lidar() 
        
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        if eval:
            reaching_goal = goal_dist < (self.robot.radius - 0.0)
        else:
            reaching_goal = goal_dist < (self.robot.radius - 0.1)

        # collision detection between the robot and humans
        collision = False
        dmin = self.scan_current.min()
        if dmin <= self.robot.radius:
            collision = True
            
        out_area = fabs(self.robot.py) > self.y_range or fabs(self.robot.px) > self.x_range
        reward = 0
        
        if ((dmin - self.robot.radius) < self.discomfort_dist):
            dis_obstacle_reward = (dmin - self.robot.radius - self.discomfort_dist) * self.discomfort_penalty_factor
        else:
            dis_obstacle_reward = 0.0

        dis_goal_reward = self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        # dis_goal_reward = 0.0
        self.goal_distance_last = goal_dist
        
        linear_reward = 0.0

        if self.action_penalty and self.quadruped_env is not None:
            if action[0] < 0.3:  # should be larger than 0.3
                linear_reward = self.linear_penalty

        rotation_penalty = 0.0
        if self.penalty_rotation and self.quadruped_env is not None:
            rotation_penalty = fabs(action_after_clip[1] - self.action_last[1]) * 0.01
            
        if self.robot_model == 'quadruped':
            self.action_last = np.array(action_after_clip, dtype=np.float32)
        else:
            self.action_last = action
        reward = dis_obstacle_reward + dis_goal_reward + linear_reward + rotation_penalty
        if collision or out_area or done_quadruped:
            reward = self.collision_penalty
            done = True
            if collision:
                info = Collision()
            elif done_quadruped:
                info = Fall()
            else:
                info = OutArea()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif ((dmin - self.robot.radius) < self.discomfort_dist):
            done = False
            info = Danger(dmin)
        else:
            done = False
            info = Nothing()
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                # if i < self.human_num - self.add_human_num:
                #     self.humans[i].gx = -self.humans[i].gx
                #     self.humans[i].gy = -self.humans[i].gy
                # else:
                #     self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)

        # get the observation
        robot_goal_state = np.array([r / self.square_width, t / np.pi, 
                                     self.action_last[0] / self.action_range[1, 0],
                                     self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
        
        self.global_step += 1
        if save_data:
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, action[0], action[1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 3), dtype=np.float32)
            for i in range(self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers

        if self.robot_policy == 'td3_lstm':
            return self.scan_current / self.laser_max_range, robot_goal_state, reward, done, info
        else:
            return self.lidar_obs / self.laser_max_range, terrain, robot_goal_state, reward, done, info
    
    def reset(self, test=False, save_data=False, seed=0):
        if test: # make test environment same for all baselines
            np.random.seed(seed)
        self.global_time = 0.0
        self.global_step = 0
        self.action_last = np.zeros(2)
        self.static_obstacles = None
        self.log_env = {}

        if self.quadruped_env is not None:
            with torch.inference_mode():
                if test:
                    quadruped_obs = self.quadruped_env.reset(seed=seed, commands=[0.0, 0.0, 0.0])
                else:
                    quadruped_obs = self.quadruped_env.reset(commands=[0.0, 0.0, 0.0])
            obs, history_obs = quadruped_obs['obs'].detach(), quadruped_obs['history_obs'].detach()
            for quadruped_step in range(100):
                with torch.inference_mode():
                    quadruped_action, encoder = self.quadruped_policy(obs, history_obs)
                    obs_dict, _, _, _ = self.quadruped_env.step(quadruped_action, commands=[0.0, 0.0, 0.0])
                    obs, history_obs = obs_dict['obs'].detach(), obs_dict['history_obs'].detach()
                    encoder = encoder.detach()
            self.specific_quadruped_envs_idx = np.random.choice(self.quadruped_num_env, size=1)[0]
            # get xy and yaw in world frame
            quadruped_pose_w = self.quadruped_env.unwrapped.get_robot_pose(self.specific_quadruped_envs_idx)
            init_x = quadruped_pose_w[0] + self.circle_radius * cos(quadruped_pose_w[2])
            init_y = quadruped_pose_w[1] + self.circle_radius * sin(quadruped_pose_w[2])
            self.init_quadruped_pose_w = np.array([init_x, init_y, quadruped_pose_w[2]], dtype=np.float32)

            if self.terrain_dim == 64:
                obs_clone = obs[self.specific_quadruped_envs_idx].detach().clone().cpu().numpy()
                encoder_clone = encoder[self.specific_quadruped_envs_idx].cpu().numpy()
                terrain = np.concatenate((obs_clone, encoder_clone)).flatten()
            else:
                terrain = None
        else:
            if self.terrain_dim > 0:
                terrain = np.zeros(self.terrain_dim, dtype=np.float32)
            else:
                terrain = None
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)
        
        self.goal_distance_last = self.robot.get_goal_distance()

        # 3,5 save
        # np.random.seed(5)

        self.generate_random_static_obstacle()
        self.generate_random_human_position()

        self.get_lidar(isReset=True) 

        # get the observation

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)
        
        robot_goal_state = np.array([r / self.square_width, t / np.pi, 
                                     self.action_last[0] / self.action_range[1, 0],
                                     self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
            
        if save_data:
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            self.log_env['goal'] = -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 3), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)
            self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

           
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 3), dtype=np.float32)
            for i in range(self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers

        if self.robot_policy == 'td3_lstm':
            return self.scan_current / self.laser_max_range, robot_goal_state
        else:
            return self.lidar_obs / self.laser_max_range, terrain, robot_goal_state

    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-5.0, 5.0)
            self.ax.set_ylim(-5.0, 5.0)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            for i in range(self.static_obstacle_num):
                self.ax.add_artist(plt.Circle((self.static_obstacles[i, 0], self.static_obstacles[i, 1]), 
                                              self.static_obstacles[i, 2],
                                              fill=True, color='c'))
            plt.text(-4.5, -4.5, str(round(self.global_time, 2)), fontsize=20)
            x, y, theta = self.robot.px, self.robot.py, self.robot.theta
            dx = cos(theta)
            dy = sin(theta)
            self.ax.arrow(x, y, dx, dy,
                width=0.01,
                length_includes_head=True, 
                head_width=0.15,
                head_length=1,
                fc='r',
                ec='r')
            ii = 0
            lines = []
            while ii < self.n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.draw()
            plt.pause(0.001)
            plt.cla()


