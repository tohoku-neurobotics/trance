import torch
import numpy as np
import time

from threading import Lock
import torch
from on_policy_runner_hardware_go2 import OnPolicyRunner

import psutil
import os
import rospy
from float_array_timestamp import FloatArrayTimestamp

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_,unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

class Go2Config:
    action_scale = 0.25

    # Velocity command limit
    max_vel_x = 0.5
    max_vel_y = 0.2
    max_vel_z = 1.0

    PosStopF = 2.146e9
    VelStopF = 16000.0

    lowcmd_topic = "rt/lowcmd"
    lowstate_topic = "rt/lowstate" 

class LocoEnv:
    def __init__(self, net="enx9c69d342225b"):
        # choose cpu or gpu
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        self.terrain_dim = 64  # 45 + 19, please revise it according your locomotion policy
        self.num_actions = 12
        self.history_len = 10
        self.num_obs = 45
        self.num_privileged_obs = 286   


        self.encoder = np.zeros(19, dtype=np.float32)
        self.obs = torch.zeros(1, self.num_obs, device=self.device, dtype=torch.float)
        self.history_obs = torch.zeros(1, self.history_len * self.num_obs, device=self.device, dtype=torch.float)

        self.cfg = Go2Config()
        self.loco_dt = 0.02
        self.user_command = np.zeros(3, dtype=np.float32)
        self.damping = False

        # load locomotion policy
        self.loco_policy = self.load_locomotion_policy()

        self.actions = np.zeros(12, dtype=np.float32)
        self.joint_position_targets = np.zeros_like(self.actions)

        self.joint_obs_order_map = [
            3, 0, 9, 6,
            4, 1, 10, 7,
            5, 2, 11, 8
        ]

        self.joint_cmd_order_map = [
            1, 5, 9,
            0, 4, 8,
            3, 7, 11,
            2, 6, 10
        ]

        # In real joint order
        self.default_joint_positions = np.array([
            -0.1, 0.8, -1.5,  # FR
            0.1, 0.8, -1.5,  # FL
            -0.1, 1.0, -1.5,  # RR
            0.1, 1.0, -1.5,  # RL
        ], dtype=np.float32)

        self.state_lock = Lock()
        self.proprioceptive_state = np.zeros(31, dtype=np.float32)

        ChannelFactoryInitialize(0, net)
        self.crc = CRC()

        self.user_command_lock = Lock()
        self.sub_user_command = rospy.Subscriber('/user_command', FloatArrayTimestamp, self.callback_user_command)
        self.pub_terrain = rospy.Publisher('/terrain', FloatArrayTimestamp, queue_size=10)

    def load_locomotion_policy(self):
        runner = OnPolicyRunner(self.num_privileged_obs, self.num_obs, 
                                self.num_actions, self.history_len, self.device)
        # runner.load('./original_reward_with_more_random/model_2200.pt')
        runner.load('./original_reward_with_more_random/2025-12-05_15-01-56_dream_waq_no_y/model_5800.pt')
        # obtain the trained policy for inference
        locomotion_policy = runner.get_inference_policy_encoder(device=self.device)
        
        t1 = time.time()
        print('start to warm up locomotion policy')
        for _ in range(10):
            t11 = time.time()
            with torch.inference_mode():
                locomotion_policy(self.obs, self.history_obs)
            t22 = time.time()
            print('locomotion inference time: ', t22 - t11)
        t2 = time.time()
        print('locomotion policy warm-up finished, time duration: ', t2 - t1)

        return locomotion_policy    
    
    def callback_user_command(self, msg):
        self.user_command_lock.acquire()
        if msg.data[0] < 0.0:
            self.user_command = np.zeros(3, dtype=np.float32)
            self.damping = True
        else:
            self.user_command[0] = msg.data[0]
            self.user_command[1] = msg.data[1]
            self.user_command[2] = msg.data[2]
        self.user_command_lock.release()
        
    def Init(self):
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher = ChannelPublisher(self.cfg.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(self.cfg.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        # wait for low_level state
        while True:
            if self.low_state is not None and self.low_state.tick != 0:
                break
            time.sleep(self.loco_dt)
        print('Successfully connected to the robot')

        # initialize low_level cmd
        self.InitLowCmd()

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def LowStateHandler(self, msg:LowStateGo):
        self.low_state = msg
        self.state_lock.acquire()

        self.proprioceptive_state[0] = msg.imu_state.gyroscope[0]
        self.proprioceptive_state[1] = msg.imu_state.gyroscope[1]
        self.proprioceptive_state[2] = msg.imu_state.gyroscope[2]

        self.proprioceptive_state[3] = msg.imu_state.quaternion[0]
        self.proprioceptive_state[4] = msg.imu_state.quaternion[1]
        self.proprioceptive_state[5] = msg.imu_state.quaternion[2]
        self.proprioceptive_state[6] = msg.imu_state.quaternion[3]

        for i in range(12):
            self.proprioceptive_state[i + 7] = msg.motor_state[i].q
            self.proprioceptive_state[i + 19] = msg.motor_state[i].dq

        self.state_lock.release()

    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF  # low level
        self.low_cmd.gpio = 0
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = self.cfg.PosStopF
            self.low_cmd.motor_cmd[i].dq = self.cfg.VelStopF
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0  
            self.low_cmd.motor_cmd[i].tau = 0

    def zero_torque_state(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0  
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.send_cmd()
        time.sleep(self.loco_dt)

    def move_to_default_pos(self):
        print('Moving to default pos.')
        t1 = time.time()
        total_time = 2
        num_step = int(total_time / self.loco_dt)

        init_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.proprioceptive_state[i + 7]

        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + self.default_joint_positions[j] * alpha
                self.low_cmd.motor_cmd[j].dq = 0.0  # qd
                self.low_cmd.motor_cmd[j].kp = 40.0
                self.low_cmd.motor_cmd[j].kd = 0.6
                self.low_cmd.motor_cmd[j].tau = 0.0
            self.send_cmd()
            time.sleep(self.loco_dt)
        t2 = time.time()
        print('time for moving to default pos: ', t2 - t1)

    def keep_default_pos_state(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.default_joint_positions[i]
            self.low_cmd.motor_cmd[i].dq = 0.0  # qd
            self.low_cmd.motor_cmd[i].kp = 40.0
            self.low_cmd.motor_cmd[i].kd = 0.6
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd()
        time.sleep(self.loco_dt)
    
    def reset_loco_env(self):  
        # reset the environment
        base_angular_vel = self.proprioceptive_state[:3]
        base_quaternion = self.proprioceptive_state[3:7]
        joint_q_real = self.proprioceptive_state[7:19]
        joint_dq_real = self.proprioceptive_state[19:31]

        # Process data
        projected_gravity = self.get_gravity_orientation(base_quaternion)
        joint_q_real = joint_q_real - self.default_joint_positions

        # Convert joint and leg orders, observations has to be in simulation order
        joint_q_isaac = np.array([joint_q_real[i] for i in self.joint_obs_order_map], dtype=np.float32)
        joint_dq_isaac = np.array([joint_dq_real[i] for i in self.joint_obs_order_map], dtype=np.float32)

        obs = np.concatenate((
            base_angular_vel,
            projected_gravity,
            self.user_command,
            joint_q_isaac,
            joint_dq_isaac,
            self.actions,
        ))

        self.encoder = np.zeros(19, dtype=np.float32)
        self.history_obs = torch.zeros(1, self.history_len * self.num_obs, device=self.device, dtype=torch.float)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            obs = obs.astype(np.float32)
        self.obs = torch.from_numpy(obs).to(self.device)

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3, dtype=np.float32)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
        
    def get_observations(self, commands):
        base_angular_vel = self.proprioceptive_state[:3]
        base_quaternion = self.proprioceptive_state[3:7]
        joint_q_real = self.proprioceptive_state[7:19]
        joint_dq_real = self.proprioceptive_state[19:31]

        # Process data
        projected_gravity = self.get_gravity_orientation(base_quaternion)
        joint_q_real = joint_q_real - self.default_joint_positions

        # Convert joint and leg orders, observations has to be in simulation order
        joint_q_isaac = np.array([joint_q_real[i] for i in self.joint_obs_order_map], dtype=np.float32)
        joint_dq_isaac = np.array([joint_dq_real[i] for i in self.joint_obs_order_map], dtype=np.float32)

        obs = np.concatenate((
            base_angular_vel,
            projected_gravity,
            commands,
            joint_q_isaac,
            joint_dq_isaac,
            self.actions,
        ))

        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            obs = obs.astype(np.float32)
        self.obs = torch.from_numpy(obs).to(self.device)
        history_obs_tmp = self.history_obs[:, self.num_obs:]
        self.history_obs = torch.cat((history_obs_tmp, self.obs), dim=-1)
        return self.obs, self.history_obs
    
    def pub_terrain_func(self):
        obs_clone = self.obs.clone()
        obs_clone = obs_clone.cpu().data.numpy().flatten()
        obs_encoder = np.concatenate((obs_clone, self.encoder))

        obs_encoder_for_pub = obs_encoder.flatten()
        obs_encoder_for_pub = obs_encoder_for_pub.tolist()
        obs_encoder_msg = FloatArrayTimestamp()
        obs_encoder_msg.header.stamp = rospy.Time.now()
        obs_encoder_msg.header.frame_id = "camera_init"   # optional
        obs_encoder_msg.data = obs_encoder_for_pub
        self.pub_terrain.publish(obs_encoder_msg)

    def rl_policy_cmd(self, actions, encoder):
        # Run inference
        self.actions = actions.squeeze(0).detach().cpu().numpy()

        # Map action to joint position targets
        joint_positions = self.actions * self.cfg.action_scale

        # Convert to real joint order and offset by default joint positions
        self.joint_position_targets = np.array([joint_positions[i] for i in self.joint_cmd_order_map], dtype=np.float32)
        self.joint_position_targets += self.default_joint_positions
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.joint_position_targets[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 20.0
            self.low_cmd.motor_cmd[i].kd = 0.5
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd()

        encoder = encoder.detach()
        self.encoder = encoder.cpu().data.numpy().flatten()
    
    def damping_cmd(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 8.0  
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd()

    def send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def locomotion_thread(self):
        count = 0
        while not rospy.is_shutdown():
            if self.damping:
                print('damping mode and exit locomotion')
                self.damping_cmd()
                break
            t_start = time.time()  
            with torch.inference_mode():
                actions, encoder = self.loco_policy(self.obs, self.history_obs)
                self.rl_policy_cmd(actions, encoder)
            t1 = time.time()
            if count % 10 == 0:
                print('locomotion inference time: ',t1 - t_start)
            count = count + 1 
            t_end = time.time()
            elapsed_time = t_end - t_start
            if elapsed_time < self.loco_dt:
                time.sleep(self.loco_dt - elapsed_time) 
            self.get_observations(self.user_command)
            self.pub_terrain_func()
            
def main(): 
    p = psutil.Process(os.getpid())
    p.cpu_affinity([7])
    print("Using CPU cores:", p.cpu_affinity())

    print('start Go2 locomotion node')
    rospy.init_node("locomotion", anonymous=True)
    locomotion_env_node = LocoEnv()
    try:
        locomotion_env_node.Init()
        time.sleep(1.0)
        locomotion_env_node.zero_torque_state()
        time.sleep(1.0)
        locomotion_env_node.move_to_default_pos()
        for i in range(250):  # keep 5 seconds
            locomotion_env_node.keep_default_pos_state()
        locomotion_env_node.reset_loco_env()
        locomotion_env_node.locomotion_thread()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.signal_shutdown("Condition met, shutting down.")

if __name__ == "__main__":
    main()