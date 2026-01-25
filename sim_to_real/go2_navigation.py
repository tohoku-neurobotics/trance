import numpy as np
import os
import torch
import rospy
import time
from sac import SacAeAgent


from pynput import keyboard

import psutil
import rospy
from float_array_timestamp import FloatArrayTimestamp
from message_filters import Subscriber, ApproximateTimeSynchronizer

terrain_dim = 64
encoder_feature_dim = 50

class Go2Navigation:
    def __init__(self):
        # choose cpu or gpu
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        self.nav_dt = 0.2
        self.frame_stack = 6
        self.square_width = 10.0
        self.n_laser = 1800
        self.robot_goal_state_dim = 4
        self.last_action = np.zeros(2, dtype=np.float32)
        self.action_dim = 2
        self.action_range = np.array([[0.0, -1.0],
                                      [0.5,  1.0]])

        # load navigation policy
        self.navigation_policy = self.load_navigation_policy()
        
        # keyboard control
        self.damping = False
        self.rl_nav = True
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.pub_user_command = rospy.Publisher('/user_command', FloatArrayTimestamp, queue_size=10)

        # Create message_filters subscribers (NOT rospy.Subscriber)
        self.sub_feature_goal = Subscriber("/feature_goal", FloatArrayTimestamp)
        self.sub_terrain = Subscriber("/terrain", FloatArrayTimestamp)

        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_feature_goal, self.sub_terrain],
            queue_size=10,
            slop=self.nav_dt  # seconds of allowable time difference
        )
        self.ts.registerCallback(self.sync_callback)

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
        feature = np.zeros(encoder_feature_dim, dtype=np.float32)
        terrain = np.zeros(terrain_dim, dtype=np.float32)
        robot_goal_state = np.zeros(self.robot_goal_state_dim, dtype=np.float32)
        t1 = time.time()
        for _ in range(10):
            t11 = time.time()
            with torch.inference_mode():
                navigation_policy.select_action(feature, terrain, robot_goal_state)
            t22 = time.time()
            print('navigation inference time: ', t22 - t11)
        t2 = time.time()
        print('navigation policy warm-up finished, time duration: ', t2 - t1)

        return navigation_policy
    
    def pub_user_command_func(self, navigation_action, damping=False):
        if damping:
            user_command = [-0.1, 0.0, 0.0] # for damping
        else:
            user_command = [navigation_action[0], 0.0, navigation_action[1]]
        user_command_msg = FloatArrayTimestamp()
        user_command_msg.header.stamp = rospy.Time.now()
        user_command_msg.header.frame_id = "camera_init"   # optional
        user_command_msg.data = user_command
        self.pub_user_command.publish(user_command_msg)

    def sync_callback(self, feature_goal_msg, terrain_msg):
        feature = np.array(feature_goal_msg.data[:-2], dtype=np.float32)
        terrain = np.array(terrain_msg.data, dtype=np.float32)
        robot_goal_state = np.array([feature_goal_msg.data[-2],
                                     feature_goal_msg.data[-1],
                                     self.last_action[0] / self.action_range[1, 0],
                                     self.last_action[1] / self.action_range[1, 1]], dtype=np.float32)
        t1 = time.time()
        if robot_goal_state[0] * self.square_width < 0.3:
            navigation_action = np.zeros(2)
        else:
            with torch.inference_mode():
                navigation_action = self.navigation_policy.select_action(feature, terrain, robot_goal_state)
        # navigation_action = np.zeros(2)  # only for test without moving
        if navigation_action[0] < 0.3:  # clip action
            navigation_action = [0.0, 0.0, 0.0]
        self.pub_user_command_func(navigation_action)
        print('navigation action: ', navigation_action)
        
        t2 = time.time()
        print('navigation inference time: ', t2 - t1)
        self.last_action = navigation_action

    def planning_loop(self):
        while self.rl_nav and (not rospy.is_shutdown()):
            try:
                time.sleep(self.nav_dt) 
            except KeyboardInterrupt:
                self.rl_nav = False
                self.damping = True
                for i in range(5):
                    self.pub_user_command_func([-0.1, 0.0, 0.0], damping=True)
                    time.sleep(0.1)
                break
            
        for i in range(5):
            self.pub_user_command_func([-0.1, 0.0, 0.0], damping=True)
            time.sleep(0.1)

    def on_press(self, key):
        try:
            if key.char == 'd':
                print(' key received, start damping mode')
                self.damping = True
                self.rl_nav = False

        except AttributeError:
            print('keyboard error or no key pressed')

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.cpu_affinity([0])  
    print("Using CPU cores:", p.cpu_affinity())

    rospy.init_node("go2_navigation", anonymous=True)
    node = Go2Navigation()

    node.planning_loop()
