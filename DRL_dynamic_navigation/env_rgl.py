import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import numpy as np

from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import ObservableState, JointState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
from collections import deque

class CrowdSim:
    def __init__(self, args, action_range, action_choices=None):
        self.square_width = args.square_width
        self.human_num_max = args.human_num_max
        self.static_obstacle_num_max = args.static_obstacle_num_max
        self.use_angular = args.use_angular
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        
        # last-time distance from the robot to the goal
        self.goal_distance_last = None
        
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

        # robot orca policy
        self.robot_policy = policy_factory['orca']()
        self.robot_policy.time_step = self.time_step
        self.robot_policy.max_speed = self.robot.v_pref
        self.robot_policy.radius = self.robot.radius
        self.robot_policy.max_robot_speed = self.robot.v_pref

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
                    if hypot(positions_x[i] - positions_x[j], positions_y[i] - positions_y[j]) <= radiuses[i] + radiuses[j] - 0.1:
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
    
    def cal_orca_action(self):
        # observation for humans is always coordinates
        ob = [human.get_observable_state() for human in self.humans]
        for k in range(self.static_obstacle_num):
            ob.append(ObservableState(
                    self.static_obstacles[k, 0], 
                    self.static_obstacles[k, 1], 
                    0.0, 0.0, self.static_obstacles[k, 2])
                    )
        state = self.robot.get_joint_state(ob)
        action_temp = self.robot_policy.predict(state)
        linear_v = hypot(action_temp[0], action_temp[1])
        target_theta = atan2(action_temp[1], action_temp[0])
        delta_theta = target_theta - self.robot.theta
        if delta_theta > np.pi:
            delta_theta -= (2.0 * np.pi)
        elif delta_theta < -np.pi:
            delta_theta += (2.0 * np.pi)
        angular_v = delta_theta / self.time_step
        action = np.array([linear_v, angular_v], dtype=np.float32)
        return action

    def get_ob(self):
        ob = []
        for k in range(self.human_num):
            ob.append(self.humans[k].get_observable_state())
        for i in range(self.human_num_max - self.human_num):
            ob.append(ObservableState(0.0, 0.0, 0.0, 0.0, 0.0))
        for k in range(self.static_obstacle_num):
            ob.append(ObservableState(self.static_obstacles[k, 0], 
                                      self.static_obstacles[k, 1], 
                                      0.0, 0.0, 
                                      self.static_obstacles[k, 2]))
        for i in range(self.static_obstacle_num_max - self.static_obstacle_num):
            ob.append(ObservableState(0.0, 0.0, 0.0, 0.0, 0.0))
        return ob
    
    def get_min_dis(self):
        min_dis = 9999.9
        for k in range(self.human_num):
            dis = hypot(self.robot.px - self.humans[k].px, self.robot.py - self.humans[k].py) \
                  - self.robot.radius - self.humans[k].radius
            min_dis = min(dis, min_dis)
        for k in range(self.static_obstacle_num):
            dis = hypot(self.robot.px - self.static_obstacles[k, 0], self.robot.py - self.static_obstacles[k, 1]) \
                  - self.robot.radius - self.static_obstacles[k, 2]
            min_dis = min(dis, min_dis)
        return min_dis
        
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

        self.global_time += self.time_step

        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        reaching_goal = goal_dist < self.robot.radius

        # collision detection between the robot and humans
        collision = False
        dmin = self.get_min_dis()
        if dmin <= 0.0:
            collision = True
            
        out_area = fabs(self.robot.py) > self.y_range or fabs(self.robot.px) > self.x_range
        reward = 0
        
        if (dmin < self.discomfort_dist):
            dis_obstacle_reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor
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
        elif (dmin < self.discomfort_dist):
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
        
        if save_data:
            self.global_step += 1
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, action[0], action[1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_info = np.zeros((self.human_num, 3), dtype=np.float32)
            for i in range (self.human_num):
                humans_info[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_info
            static_obstacles_info = np.zeros((self.static_obstacle_num, 3), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles[i, 0], 
                                                     self.static_obstacles[i, 1],
                                                     self.static_obstacles[i, 2]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info

        ob = self.get_ob()
        state = self.robot.get_joint_state(ob)

        return state, reward, done, info
    
    def reset(self, seed=-1, save_data=False):
        self.global_time = 0.0
        self.global_step = 0
        self.action_last = np.zeros(2)
        self.static_obstacles = None
        self.log_env = {}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)
        
        self.goal_distance_last = self.robot.get_goal_distance()

        # 3,5 save
        # np.random.seed(5)
        if seed >= 0:
            np.random.seed(seed)
        self.generate_random_static_obstacle()
        self.generate_random_human_position()
       
        if save_data:
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 3), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 3), dtype=np.float32)

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

        ob = self.get_ob()
        state = self.robot.get_joint_state(ob)

        return state




