#!/usr/bin/env python

import rospy
import heapq
from float_array_timestamp import FloatArrayTimestamp
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
from threading import Lock
from math import fabs, hypot, cos, sin, atan2
import time
from datetime import datetime
import os
from sac import SacAeAgent
import torch
from lidar_reconstruction.motion_plan_lib_opt import *

import psutil

terrain_dim = 64

class NavEnv:
    def __init__(self):        

        self.nav_dt = 0.2
        # create log folder and file
        file_evaluation_episodes = './evaluation_episodes'
        if not os.path.exists(file_evaluation_episodes):
            os.makedirs(file_evaluation_episodes)
        # Get current date and time
        now = datetime.now()
        # Format: YYYYMMDD_HHMMSS
        self.log_filename_prefix = file_evaluation_episodes + '/log_' + now.strftime("%Y%m%d_%H%M%S")

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        self.damping = False
        self.action_dim = 2
        self.frame_stack = 6
        self.n_laser = 1800
        self.robot_goal_state_dim = 4 
        self.action_range = np.array([[0.0, -1.0],
                                      [0.5,  1.0]])
        # load navigation policy
        self.navigation_policy = self.load_navigation_policy()

        # Create message_filters subscribers (NOT rospy.Subscriber)
        self.sub_humans = Subscriber("/pedestrians", FloatArrayTimestamp)
        self.sub_robot_pose_2d = Subscriber("/robot/pose2d", FloatArrayTimestamp)
        self.sub_user_command = rospy.Subscriber('/user_command', FloatArrayTimestamp, self.callback_user_command)

        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_humans, self.sub_robot_pose_2d],
            queue_size=10,
            slop=self.nav_dt  # seconds of allowable time difference
        )
        self.ts.registerCallback(self.sync_callback)
        self.pub_feature_goal = rospy.Publisher('/feature_goal', FloatArrayTimestamp, queue_size=10)
        
        # 
        self.lock_pedestrians = Lock()
        self.pedestrians = []
        self.static_pedestrians = {}
        self.camera_to_lidar = [-0.1, 0.0]

        # used to record robot pose
        self.lock_robot_pose = Lock()
        self.robot_pose = np.zeros(3)
        self.robot_pose_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # LiDAR scan environment
        
        self.human_radius = 0.3
        
        self.laser_angle_resolute = 0.003490659
        self.laser_min_range = 0.27
        self.laser_max_range = 6.0
        self.square_width = 10.0
        self.goal = [6.0, 0.0]  # fixed goal
        self.max_human_num = 7
        self.human_dim = 4
    
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32) # used for visualization

        # laser state
        self.scan_current = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        self.scan_end_current = np.zeros((self.n_laser, 2), dtype=np.float32)

        self.scan_last = np.zeros((self.frame_stack - 1, self.n_laser), dtype=np.float32)
        self.scan_end_last = np.zeros((self.frame_stack - 1, self.n_laser, 2), dtype=np.float32)
        self.lidar_obs = np.zeros(1 * self.frame_stack * self.n_laser, dtype=np.float32)

        # log lidar, robot, and humans
        self.global_step = 0
        self.max_episode_step = 150
        self.log_env = {}

    def load_navigation_policy(self):
        obs_shape = (1, self.frame_stack, self.n_laser)
        navigation_policy = SacAeAgent(
                                obs_shape,
                                terrain_dim,
                                self.robot_goal_state_dim,
                                self.action_dim,
                                self.action_range,
                                self.device
                            )
        navigation_policy.load('./navigation_network/step_1200000', device=self.device)

        print('start to warm up navigation policy')
        lidar_image = np.zeros((1, self.frame_stack, self.n_laser), dtype=np.float32)
        t1 = time.time()
        for _ in range(10):
            t11 = time.time()
            with torch.inference_mode():
                navigation_policy.get_features(lidar_image)
            t22 = time.time()
            print('navigation inference time: ', t22 - t11)
        t2 = time.time()
        print('navigation policy warm-up finished, time duration: ', t2 - t1)

        return navigation_policy


    def sync_callback(self, humans_msg, robot_pose_msg):
        self.lock_pedestrians.acquire()
        self.pedestrians = []
        for i in range(self.max_human_num):
            if fabs(humans_msg.data[self.human_dim * i + 0]) > 50.0:
                break
            else:
                # id, x, y, moving
                self.pedestrians.append(np.array([humans_msg.data[self.human_dim * i + 0],
                                                  humans_msg.data[self.human_dim * i + 1],
                                                  humans_msg.data[self.human_dim * i + 2],
                                                  humans_msg.data[self.human_dim * i + 3]], dtype=np.float32))
        self.lock_pedestrians.release()
        
        self.lock_robot_pose.acquire()
        # x, y, yaw
        self.robot_pose[0] = robot_pose_msg.data[0]
        self.robot_pose[1] = robot_pose_msg.data[1]
        self.robot_pose[2] = robot_pose_msg.data[2]
        self.lock_robot_pose.release()

    def callback_user_command(self, msg):
        if msg.data[0] < 0.0:
            self.damping = True

    '''use c library to generate lidar scan, which is fast'''
    def get_lidar(self, robot_pose_current_offset, pedestrians_current_transform, isReset=False):
        num_circle_human = len(pedestrians_current_transform)
        robot_pose_new = np.array([robot_pose_current_offset[0], 
                                   robot_pose_current_offset[1], 
                                   robot_pose_current_offset[2]], dtype=np.float32)
        human_circles = -100.0 * np.ones((self.max_human_num, 3), dtype=np.float32)
        for i in range(num_circle_human):
            human_circles[i] = np.array([pedestrians_current_transform[i][0], 
                                         pedestrians_current_transform[i][1], 
                                         self.human_radius], dtype=np.float32)
        self.log_env['humans'][self.global_step % self.max_episode_step] = human_circles
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
            
        lidar_obs = np.hstack((self.scan_current, self.scan_last.flatten()))
        self.lidar_obs = lidar_obs.flatten()

    def get_robot_goal_state(self, robot_pose_current_offset):
        dx = self.goal[0] - robot_pose_current_offset[0]
        dy = self.goal[1] - robot_pose_current_offset[1]
        theta = robot_pose_current_offset[2]
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)
        
        robot_goal_state = np.array([r / self.square_width, t / np.pi], dtype=np.float32)
        
        return robot_goal_state
    
    def transform_pedestrian_into_global_frame(self, robot_pose_current, pedestrians_current):
        all_pedestrians = []
        # version 1
        # for ped in pedestrians_current:
        #     if fabs(ped[0]) > 50.0:
        #         break
        #     id = str(ped[0])
        #     x_lidar = ped[1] + self.camera_to_lidar[0]
        #     y_lidar = ped[2] + self.camera_to_lidar[1] 
        #     s_theta = sin(robot_pose_current[2])
        #     c_theta = cos(robot_pose_current[2])
        #     x_global = x_lidar * c_theta - y_lidar * s_theta + robot_pose_current[0] + self.robot_pose_offset[0]
        #     y_global = x_lidar * s_theta + y_lidar * c_theta + robot_pose_current[1] + self.robot_pose_offset[1]
        #     moving = ped[3]
        #     if moving < 0.5:
        #         for id_static in self.static_pedestrians.keys():
        #             dis = hypot(x_global - self.static_pedestrians[id_static][0], y_global - self.static_pedestrians[id_static][1])
        #             if dis < 0.15:
        #                 del self.static_pedestrians[id_static]
        #                 break
        #         self.static_pedestrians[id] = np.array([x_global, y_global], dtype=np.float32)
        #     else:
        #         if id in self.static_pedestrians:
        #             del self.static_pedestrians[id]
        #         all_pedestrians.append(np.array([x_global, y_global], dtype=np.float32))
        # for value in self.static_pedestrians.values():
        #     all_pedestrians.append(value)
        # version 1

        for ped in pedestrians_current:
            if fabs(ped[0]) > 50.0:
                break
            x_lidar = ped[1] + self.camera_to_lidar[0]
            y_lidar = ped[2] + self.camera_to_lidar[1] 
            s_theta = sin(robot_pose_current[2])
            c_theta = cos(robot_pose_current[2])
            x_global = x_lidar * c_theta - y_lidar * s_theta + robot_pose_current[0] + self.robot_pose_offset[0]
            y_global = x_lidar * s_theta + y_lidar * c_theta + robot_pose_current[1] + self.robot_pose_offset[1]
            all_pedestrians.append(np.array([x_global, y_global], dtype=np.float32))

        pedestrians_current_transform = []
        if len(all_pedestrians) > self.max_human_num:
            dis_all = []
            for ped in all_pedestrians:
                dis = hypot(ped[0], ped[1])
                dis_all.append(dis)
            # Find the self.max_human_num smallest values and their indexes
            smallest = heapq.nsmallest(self.max_human_num, enumerate(dis_all), key=lambda x: x[1])
            # Separate indexes and values
            indexes = [i for i, v in smallest]
            pedestrians_current_transform= [all_pedestrians[j] for j in indexes]
        else:
            pedestrians_current_transform = all_pedestrians
        return pedestrians_current_transform

    def get_obs(self):
        robot_pose_current = self.robot_pose
        robot_pose_current_offset = robot_pose_current + self.robot_pose_offset
        pedestrians_current = self.pedestrians
        pedestrians_current_transform = self.transform_pedestrian_into_global_frame(robot_pose_current, pedestrians_current)
        self.get_lidar(robot_pose_current_offset, pedestrians_current_transform, self.global_step == 0)
        robot_goal_state = self.get_robot_goal_state(robot_pose_current_offset)
        
        self.log_env['robot'][self.global_step % self.max_episode_step] = np.array([robot_pose_current_offset[0], 
                                                            robot_pose_current_offset[1], 
                                                            robot_pose_current_offset[2]])
        self.log_env['goal'][self.global_step % self.max_episode_step] = np.array([self.goal[0], self.goal[1]])
       
        lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
        for i in range(self.n_laser):
            laser = self.scan_intersection[i]
            lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
        # self.log_env['laser'][self.global_step % self.max_episode_step] = lasers

        self.global_step = self.global_step + 1

        # if self.global_step % self.max_episode_step == 0:
        #     num = int(self.global_step / self.max_episode_step)
        #     file_name = self.log_filename_prefix + '_' + (str(num)) + '.npz'
        #     np.savez_compressed(file_name, **self.log_env)

        lidar_image = np.reshape(self.lidar_obs / self.laser_max_range, (1, self.frame_stack, self.n_laser))
        feature = self.navigation_policy.get_features(lidar_image)
        return feature, robot_goal_state

    def reset_nav_env(self):
        self.global_step = 0
        self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step, 3), dtype=np.float32)
        self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step, 2), dtype=np.float32)
        self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step, self.max_human_num, 3), dtype=np.float32)
        # self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step, self.n_laser, 4), dtype=np.float32)

    def navigation_env_thread(self):
        while not rospy.is_shutdown():
            if self.damping:
                print('dampping mode, so exit')
                self.save_trajectory()
                break
            t_start = time.time()  
            feature, robot_goal_state = self.get_obs()
            t1 = time.time()
            print('lidar construction time: ', t1 - t_start)
            t_end = time.time()
            elapsed_time = t_end - t_start
            if elapsed_time < self.nav_dt:
                time.sleep(self.nav_dt - elapsed_time) 

            obs_feature_goal = np.concatenate((feature, robot_goal_state))
            obs_feature_goal_for_pub = obs_feature_goal.flatten()
            obs_feature_goal_for_pub = obs_feature_goal_for_pub.tolist()
            obs_feature_goal_msg = FloatArrayTimestamp()
            obs_feature_goal_msg.header.stamp = rospy.Time.now()
            obs_feature_goal_msg.header.frame_id = "camera_init"   # optional
            obs_feature_goal_msg.data = obs_feature_goal_for_pub
            self.pub_feature_goal.publish(obs_feature_goal_msg)

    def save_trajectory(self):
        file_name = self.log_filename_prefix + '.npz'
        np.savez_compressed(file_name, **self.log_env)

def main(): 
    p = psutil.Process(os.getpid())
    p.cpu_affinity([4, 3, 2, 1])  
    print("Using CPU cores:", p.cpu_affinity())

    print('start Go2 navigation environment node')
    rospy.init_node("navigation_env", anonymous=True)
    navigation_env_node = NavEnv()
    try:
        navigation_env_node.reset_nav_env()
        navigation_env_node.navigation_env_thread()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.signal_shutdown("Condition met, shutting down.")

if __name__ == "__main__":
    main()
