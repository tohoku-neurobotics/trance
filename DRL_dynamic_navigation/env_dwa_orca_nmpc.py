import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import numpy as np
import casadi as ca

from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import ObservableState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
from NeuPAN.neupan.neupan import neupan

class CrowdSim:
    def __init__(self, args, action_range, action_choices=None):
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.robot_model = args.robot_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        self.human_num_max = args.human_num_max
        self.static_obstacle_num_max = args.static_obstacle_num_max
        self.use_angular = args.use_angular
        
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

        # dwa parameters
        self.acc_linear_max = self.acceleration[0]
        self.acc_angular_max = self.acceleration[1]
        self.dwa_resolution_linear_v = 0.02
        self.dwa_resolution_angular_v = 0.02
        self.dwa_look_forward_steps = 5
        self.dwa_dist_goal_cost = 0.4
        self.delta_linear_v_max = self.acc_linear_max * self.time_step
        self.delta_angular_v_max = self.acc_angular_max * self.time_step
        self.delta_linear_v = 0.05
        self.delta_angular_v = 0.05

        # robot orca policy
        if args.policy == 'orca' or args.policy == 'avocado':
            self.robot_policy = policy_factory[args.policy]()
            self.robot_policy.time_step = self.time_step
            self.robot_policy.max_speed = self.robot.v_pref
            self.robot_policy.radius = self.robot.radius
            self.robot_policy.max_robot_speed = self.robot.v_pref
        if args.policy == 'neupan':
            planner_file = './NeuPAN/example/dyna_obs/diff/planner.yaml'
            self.neupan_planner = neupan.init_from_yaml(planner_file)

        self.frame_stack = args.frame_stack

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

    def get_lidar(self):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        # robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        num_line = self.lines.shape[0]
        num_circle_human = self.human_num
        num_circle_obstacle = self.static_obstacle_num
        InitializeEnv(num_line, num_circle_human + num_circle_obstacle, self.n_laser, self.frame_stack, self.laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle_human):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            set_circles(3 * i + 2, self.humans[i].radius)
        for i in range (num_circle_obstacle):
            set_circles(3 * (i + num_circle_human)    , self.static_obstacles[i, 0])
            set_circles(3 * (i + num_circle_human) + 1, self.static_obstacles[i, 1])
            set_circles(3 * (i + num_circle_human) + 2, self.static_obstacles[i, 2])
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32)
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            ### used for visualization
            self.scan_intersection[i, 0, 0] = self.robot.px
            self.scan_intersection[i, 0, 1] = self.robot.py
            self.scan_intersection[i, 1, 0] = get_scan_line(4 * i + 2)
            self.scan_intersection[i, 1, 1] = get_scan_line(4 * i + 3)
            ### used for visualization
        self.scan_current = np.clip(scan, self.laser_min_range, self.laser_max_range)
       
        ReleaseEnv()

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
    
    def cal_avocado_action(self):
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
    
    def cal_neupan_action(self):
        robot_state = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        robot_state = robot_state[:, None]
        lidar_scan = self.scan_current[::18]  # initial lidar dim is 1800; get a new lidar with dim 100
        lidar_scan_dict = {
            'ranges': lidar_scan,
            'angle_min': -np.pi,
            'angle_max': np.pi,
            'range_min': self.laser_min_range,
            'range_max': self.laser_max_range
        }
        points = self.neupan_planner.scan_to_point(robot_state, lidar_scan_dict)
        point_velocities = None
        action, info = self.neupan_planner(robot_state, points, point_velocities)
        return action[:, 0]
    
    def cal_nmpc_action(self):
        # Robot dynamics
        def robot_dynamics(x, u):
            theta = x[2]
            v = u[0]
            w = u[1]
            return ca.vertcat(
                v * ca.cos(theta),
                v * ca.sin(theta),
                w
            )
        
        # Nonlinear MPC setup with constant-velocity pedestrian prediction
        def nmpc_setup(ped_radii, safe_margin=0.2):
            N = 10
            NX = 3
            NU = 2
            dt = self.time_step
            goal = np.array([self.robot.gx, self.robot.gy], dtype=np.float32)

            opti = ca.Opti()

            X = opti.variable(NX, N + 1)   # x, y, theta
            U = opti.variable(NU, N)       # v, w
            P = opti.parameter(NX + (self.human_num + self.static_obstacle_num) * 4)  # robot x, y, theta + ped x, y, vx, vy

            Q = np.diag([10, 10])
            R = np.diag([0.5, 0.1])

            # Dynamics constraints
            for k in range(N):
                x_next = X[:, k] + dt * robot_dynamics(X[:, k], U[:, k])
                opti.subject_to(X[:, k + 1] == x_next)

            # Cost function
            cost = 0
            for k in range(N):
                state_error = X[:2, k] - goal
                cost += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([U[:, k].T, R, U[:, k]])
            opti.minimize(cost)

            # Robot control constraints
            opti.subject_to(opti.bounded(self.action_range[0, 0], U[0, :], self.action_range[1, 0]))
            opti.subject_to(opti.bounded(self.action_range[0, 1], U[1, :], self.action_range[1, 1]))

            # Initial state constraint
            opti.subject_to(X[:, 0] == P[0:3])

            # Collision avoidance
            for k in range(N + 1):
                for i in range(self.human_num + self.static_obstacle_num):
                    base_idx = 3 + i * 4
                    px0 = P[base_idx]
                    py0 = P[base_idx + 1]
                    vx = P[base_idx + 2]
                    vy = P[base_idx + 3]
                    px_k = px0 + k * dt * vx
                    py_k = py0 + k * dt * vy

                    combined_radius = self.robot.radius + ped_radii[i] + safe_margin
                    dist = (X[0, k] - px_k)**2 + (X[1, k] - py_k)**2
                    opti.subject_to(dist >= combined_radius**2)

            opts = {"ipopt.print_level": 0, "print_time": 0}
            opti.solver("ipopt", opts)
            return opti, X, U, P
        
        ped_radii = []
        for j1 in range(self.human_num):
            ped_radii.append(self.humans[j1].radius)
        for j2 in range(self.static_obstacle_num):
            ped_radii.append(self.static_obstacles[j2, 2])

        opti, X, U, P = nmpc_setup(
            ped_radii,
            safe_margin=0.2
        )

        x0 = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        
        ped_states = []
        for j1 in range(self.human_num):
            ped_states.append(np.array([self.humans[j1].px, self.humans[j1].py, self.humans[j1].vx, self.humans[j1].vy], dtype=np.float32))
        for j2 in range(self.static_obstacle_num):
            ped_states.append(np.array([self.static_obstacles[j2, 0], self.static_obstacles[j2, 1], 0.0, 0.0], dtype=np.float32))
        flat_peds = np.array(ped_states).flatten()
        param = np.concatenate([x0, flat_peds])
        opti.set_value(P, param)
        try:
            sol = opti.solve()
            ctrl = sol.value(U[:, 0])
        except:
            # print("MPC failed to solve.")
            ctrl = np.zeros(2)
        return ctrl
    
    def cal_tmpc_action(self):
        # Robot dynamics
        def robot_dynamics(x, u):
            theta = x[2]
            v = u[0]
            w = u[1]
            return ca.vertcat(
                v * ca.cos(theta),
                v * ca.sin(theta),
                w
            )
        
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
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers

        return reward, done, info
    
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

        self.get_lidar() 
        # get the observation
   
        if save_data:
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


