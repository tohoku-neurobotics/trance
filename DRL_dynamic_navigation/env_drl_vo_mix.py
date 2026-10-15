import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import numpy as np

from numpy.linalg import norm
from utils_trans.human import Human
from utils_trans.robot import Robot
from utils_trans.state import ObservableState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib_opt import *
from collections import deque

class CrowdSim:
    def __init__(self, args, action_range, action_choices=None, human_policy_selection='mix'):
        # human policy: orca, sfm, mix (either orca or sfm)
        if (not human_policy_selection == 'orca') and (not human_policy_selection == 'sfm') and (not human_policy_selection == 'mix'):
            raise NotImplementedError(human_policy_selection)
        self.human_policy = human_policy_selection
        self.human_policy_selection = human_policy_selection

        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.robot_policy = args.policy
        self.robot_model = args.robot_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        self.human_num_max = args.human_num_max
        self.static_obstacle_num_max = args.static_obstacle_num_max
        self.use_angular = args.use_angular
        self.random_radius = args.random_radius
        self.test_single = args.test_single
        
        
        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32) # used for visualization

        # laser state
        self.scan_current = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        
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
        self.angular_penalty = -0.1

        self.static_obstacles_sfm = []
        self.human_policy_sfm = None

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
        self.static_obstacles = None

        self.human_num = None
        self.static_obstacle_num = None

        self.humans = None
        self.human_v_pref = 1.0
        self.rectangles = None
        self.action_range = action_range
        self.action_choices = action_choices
        self.robot = Robot()
        self.robot.time_step = self.time_step
        self.robot.v_pref = action_range[1, 0]
        self.action_last = np.zeros(2)
        self.acceleration = [1.0, 1.0]
        self.robot_visible_threshold = 1.5
        
        # observation image
        self.frame_stack = 10
        self.lidar_frames = deque([], maxlen=self.frame_stack)
        self.image_size = args.image_size
        self.range_resolution = self.laser_max_range / self.image_size
        self.lidar_resolution = int(self.n_laser / self.image_size)
        
        self.r_resolution = self.laser_max_range / self.image_size
        self.theta_resolution = self.n_laser / self.image_size

        # visualization on 2D plane
        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # log lidar, robot, and humans
        self.log_env = {}

    def get_agent_state(self, init_state=False):
        agent_state_list = []
        for i in range(self.human_num):
            # px, py, vx, vy, gx, gy
            single_human_state = np.array([self.humans[i].px, self.humans[i].py, 
                                           self.humans[i].vx, self.humans[i].vy,   
                                           self.humans[i].gx, self.humans[i].gy], dtype=np.float32)
            if init_state:
                single_human_state[2] = 1.0   # human preferred speed
                single_human_state[3] = 0.0
            agent_state_list.append(single_human_state)
        # assuming the robot is static, px = gx, and py = gy
        state_robot = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.px, self.robot.py], dtype=np.float32)
        agent_state_list.append(state_robot)  # the last state is from the robot
        agent_state_numpy = np.array(agent_state_list)
        return agent_state_numpy

    def generate_random_static_obstacle(self):
        self.static_obstacle_num = int(np.random.randint(self.static_obstacle_num_max, size=1)[0] + 1)
        self.static_obstacles = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
        while True:
            positions_x = np.random.uniform(-self.static_obstacle_area_x, self.static_obstacle_area_x, 
                                            (self.static_obstacle_num, 1))
            positions_y = np.random.uniform(-self.static_obstacle_area_y, self.static_obstacle_area_y, 
                                            (self.static_obstacle_num, 1))
            # radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            # keep radius constant
            if self.random_radius:
                radiuses = np.random.uniform(0.2, 0.4, (self.static_obstacle_num, 1))
            else:
                radiuses = 0.3 * np.ones((self.static_obstacle_num, 1), dtype=np.float32)
            collision = False
            for i in range(self.static_obstacle_num):
                temp = False
                for j in range(i + 1, self.static_obstacle_num):
                    # allow 0.1m overlap
                    if hypot(positions_x[i] - positions_x[j], positions_y[i] - positions_y[j]) <= radiuses[i] + radiuses[j] - 0.1:
                        collision = True
                        temp = True
                        break
                if temp:
                    break
            if not collision:
                self.static_obstacles = np.hstack((positions_x, positions_y, radiuses))
                break
        
        if self.human_policy == 'sfm':
            self.static_obstacles_sfm = []
            for i in range(self.static_obstacle_num):
                x_center = self.static_obstacles[i, 0]
                y_center = self.static_obstacles[i, 1]
                half_size = self.static_obstacles[i, 2]
                x_min = x_center - half_size
                x_max = x_center + half_size
                y_min = y_center - half_size
                y_max = y_center + half_size
                self.static_obstacles_sfm.append([x_min, x_min, y_min, y_max])
                self.static_obstacles_sfm.append([x_min, x_max, y_max, y_max])
                self.static_obstacles_sfm.append([x_max, x_max, y_min, y_max])
                self.static_obstacles_sfm.append([x_min, x_max, y_min, y_min])
        
    def generate_random_human_position(self):
        self.human_num = int(np.random.randint(self.human_num_max, size=1)[0] + 1)
        self.humans = [None] * self.human_num
        for i in range(self.human_num):
            self.humans[i] = self.generate_circle_crossing_human()

        if self.human_policy == 'orca':
            for i in range(self.human_num):
                human_policy = policy_factory['orca']()
                human_policy.time_step = self.time_step
                human_policy.max_speed = self.humans[i].v_pref
                human_policy.radius = self.humans[i].radius
                human_policy.max_robot_speed = self.robot.v_pref
                self.humans[i].set_policy(human_policy)
        elif self.human_policy == 'sfm':
            initial_state_numpy = self.get_agent_state(init_state=True)
            self.human_policy_sfm = policy_factory['sfm'](initial_state_numpy, obs=self.static_obstacles_sfm)

    def generate_circle_crossing_human(self):
        if self.static_obstacles is None:
            raise NotImplementedError(self.static_obstacles)
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            # Sample agent radius and v_pref attribute from certain distribution
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = self.human_v_pref
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

    def get_lidar(self):
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        robot_pose_new = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        human_circles = -100.0 * np.ones((self.human_num_max + self.static_obstacle_num_max, 3), dtype=np.float32)
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
    
        self.scan_current = np.clip(scans_raw, self.laser_min_range, self.laser_max_range)
        

    def construct_obs(self):
        # px, py, vx, vy
        self.humans_state_in_robot_frame = np.zeros((self.human_num+self.static_obstacle_num, 4), dtype=np.float32)
        ped_pos_map = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
        c_theta = cos(self.robot.theta)
        s_theta = sin(self.robot.theta)
        for i in range (self.human_num+self.static_obstacle_num):
            if i < self.human_num:
                px_world = self.humans[i].px
                py_world = self.humans[i].py
                vx_world = self.humans[i].vx
                vy_world = self.humans[i].vy
            else:
                px_world = self.static_obstacles[i - self.human_num, 0]
                py_world = self.static_obstacles[i - self.human_num, 1]
                vx_world = 0.0
                vy_world = 0.0
            dx = px_world - self.robot.px
            dy = py_world - self.robot.py
            px_local = dy * s_theta + dx * c_theta
            py_local = dy * c_theta - dx * s_theta
            vx_local = vy_world * s_theta + vx_world * c_theta
            vy_local = vy_world * c_theta - vx_world * s_theta
            self.humans_state_in_robot_frame[i, :] = np.array([px_local, py_local, vx_local, vy_local], dtype=np.float32)
            if fabs(px_local) < self.laser_max_range and fabs(py_local) < self.laser_max_range:
                r = int((self.laser_max_range - py_local) / self.range_resolution)
                c = int((px_local + self.laser_max_range) / self.range_resolution)
                if c < self.image_size and r < self.image_size and c >= 0 and r >= 0:
                    ped_pos_map[0, r, c] = vx_local
                    ped_pos_map[1, r, c] = vy_local

        lidar_map = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        lidar_avg = np.zeros((2 * self.frame_stack, self.image_size), dtype=np.float32)
        lidar_frames = np.concatenate(list(self.lidar_frames), axis=0)
        lidar_frames = np.reshape(lidar_frames, (self.frame_stack, -1))
        for idx, lidar in enumerate(lidar_frames):
            for j in range(self.image_size):
                lidar_avg[2 * idx, j] = np.min(lidar[j*self.lidar_resolution:(j+1)*self.lidar_resolution])
                lidar_avg[2 * idx + 1, j] = np.mean(lidar[j*self.lidar_resolution:(j+1)*self.lidar_resolution])
        # Repeat lidar_avg five times along the first axis
        lidar_avg_map = np.tile(lidar_avg, (5, 1))
        lidar_map[0, :, :] = lidar_avg_map
        obs = np.concatenate((ped_pos_map, lidar_map), axis=0)
        return obs
    
        
    def step(self, action, eval=False, save_data=False):
        if self.human_policy == 'orca':
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
        elif self.human_policy == 'sfm':
            human_actions = self.human_policy_sfm.predict()

        # update robot states
        action_copy = np.array([action[0], action[1]])
        robot_theta = self.robot.theta + action[1] * self.time_step
        if robot_theta > np.pi:
            robot_theta -= (2.0 * np.pi)
        elif robot_theta < -np.pi:
            robot_theta += (2.0 * np.pi)
        robot_x = self.robot.px + action[0] * self.time_step * cos(robot_theta)
        robot_y = self.robot.py + action[0] * self.time_step * sin(robot_theta)
        action_copy[0] = hypot(robot_y - self.robot.py, robot_x - self.robot.px) / self.time_step
            
        # update states
        self.robot.update_states(robot_x, robot_y, robot_theta, action_copy, differential=True)
       
        for i in range(self.human_num):
            self.humans[i].update_states(human_actions[i])
        # update human states for sfm
        if self.human_policy == 'sfm':
            agent_state = self.get_agent_state()
            self.human_policy_sfm.update(agent_state)

        self.get_lidar()
        
        self.lidar_frames.append(self.scan_current / self.laser_max_range)
        assert len(self.lidar_frames) == self.frame_stack
        obs_image = self.construct_obs()
        
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
        
        # angular_reward = fabs(action[1]) * self.angular_penalty
        angular_reward = 0.0

        self.action_last = action
        reward = dis_obstacle_reward + dis_goal_reward + angular_reward 
        if collision or out_area:
            reward = self.collision_penalty
            done = True
            info = Collision()
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
        
        if save_data:
            all_obstacles = np.zeros((self.human_num + self.static_obstacle_num, 2), dtype=np.float32)
            self.global_step += 1
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, action[0], action[1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 3), dtype=np.float32)
            for i in range (self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius], dtype=np.float32)
                all_obstacles[i] = np.array([self.humans[i].px, self.humans[i].py])
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
                all_obstacles[i + self.human_num] = np.array([self.static_obstacles[i, 0], self.static_obstacles[i, 1]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers
        
        return obs_image, robot_goal_state, reward, done, info
    

    
    def reset(self, seed=-1, save_data=False):
        self.global_time = 0.0
        self.global_step = 0
        self.action_last = np.zeros(2)
        self.static_obstacles = None
        self.log_env = {}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)
        
        self.goal_distance_last = self.robot.get_goal_distance()

        # human policy is not fixed, but mixed either orca or sfm
        if self.human_policy_selection == 'mix':
            if np.random.random() > 0.5:
                self.human_policy = 'orca'
            else:
                self.human_policy = 'sfm'

        # 3,5 save
        # np.random.seed(5)
        if seed >= 0:
            np.random.seed(seed)
        self.generate_random_static_obstacle()
        self.generate_random_human_position()

        self.get_lidar()

        for _ in range(self.frame_stack):
            self.lidar_frames.append(self.scan_current / self.laser_max_range)
        assert len(self.lidar_frames) == self.frame_stack
        obs_image = self.construct_obs()

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
            all_obstacles = np.zeros((self.human_num + self.static_obstacle_num, 2), dtype=np.float32)
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 3), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)
            self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 3), dtype=np.float32)
            for i in range(self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius], dtype=np.float32)
                all_obstacles[i] = np.array([self.humans[i].px, self.humans[i].py])
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
                all_obstacles[i + self.human_num] = np.array([self.static_obstacles[i, 0], self.static_obstacles[i, 1]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers
            
        return obs_image, robot_goal_state

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