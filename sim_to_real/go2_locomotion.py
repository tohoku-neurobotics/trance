import sys

import numpy as np
import time
from pynput import keyboard
from threading import Lock
import torch

from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_,unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC


from go2_env import Go2Env

from on_policy_runner_hardware_go2 import OnPolicyRunner

# please first make the robot lie down

def load_policy(num_privileged_obs, num_obs, num_actions, history_len, device):
    # 2200 best 
    # 3300, 3700,  4100
    resume_path = './original_reward_with_more_random/2025-12-05_15-01-56_dream_waq_no_y/model_5800.pt'  
    # resume_path = './original_reward_with_more_random/model_2200.pt'  
  
    runner = OnPolicyRunner(num_privileged_obs, num_obs, num_actions, history_len, device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy_encoder(device=device)
    
    return policy

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


class Custom:
    def __init__(self, control_dt):
        self.dt = control_dt
        self.cfg = Go2Config()
        self.env = Go2Env()

        # Policy and its hidden states
        self.policy = load_policy(self.env.num_privileged_obs, self.env.num_obs, 
                                  self.env.num_actions, self.env.history_len, self.env.device)

        # keyboard control
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
   
        self.standing = False
        self.damping = False
        self.rl_policy = False
        self.terminate = False

        print('start key listener')

        # Working variables
        self.commands = np.zeros(3, dtype=np.float32)

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
        self.cmd_lock = Lock()
        self.proprioceptive_state = np.zeros(31, dtype=np.float32)
        self.crc = CRC()

    # Public methods
    def Init(self):

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher = ChannelPublisher(self.cfg.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(self.cfg.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        # warm up networks
        t1 = time.time()
        print('start to warm up networks')
        for _ in range(10):
            t11 = time.time()
            with torch.inference_mode():
                self.policy(self.env.obs, self.env.history_obs)
            t22 = time.time()
            print('locomotion inference time: ', t22 - t11)
        t2 = time.time()
        print('warm up finished, time duration: ', t2 - t1)

        # wait for low_level state
        while True:
            if self.low_state is not None and self.low_state.tick != 0:
                break
            time.sleep(self.dt)
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

    def InitLowCmd(self):
        self.cmd_lock.acquire()
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
        self.cmd_lock.release()


    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while not self.standing:
            self.cmd_lock.acquire()
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0  
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.cmd_lock.release()
            self.send_cmd()
            time.sleep(self.dt)
        self.print_info()

    def move_to_default_pos(self):
        print('Moving to default pos.')
        t1 = time.time()
        total_time = 2
        num_step = int(total_time / self.dt)

        init_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.proprioceptive_state[i + 7]

        for i in range(num_step):
            alpha = i / num_step
            self.cmd_lock.acquire()
            for j in range(12):
                self.low_cmd.motor_cmd[j].q = init_dof_pos[j] * (1 - alpha) + self.default_joint_positions[j] * alpha
                self.low_cmd.motor_cmd[j].dq = 0.0  # qd
                self.low_cmd.motor_cmd[j].kp = 40.0
                self.low_cmd.motor_cmd[j].kd = 0.6
                self.low_cmd.motor_cmd[j].tau = 0.0
            self.cmd_lock.release()
            self.send_cmd()
            time.sleep(self.dt)
        t2 = time.time()
        self.print_info()
        print('time for moving to default pos: ', t2 - t1)

    def keep_default_pos_state(self):
        print("Keeping default pos state.")
        while self.standing:
            self.cmd_lock.acquire()
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.default_joint_positions[i]
                self.low_cmd.motor_cmd[i].dq = 0.0  # qd
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 0.6
                self.low_cmd.motor_cmd[i].tau = 0
            self.cmd_lock.release()
            self.send_cmd()
            time.sleep(self.dt)
        self.print_info()
        print("Exit default pos state.")

    def damping_cmd(self):
        self.cmd_lock.acquire()
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 8.0  
            self.low_cmd.motor_cmd[i].tau = 0
        self.cmd_lock.release()
        self.send_cmd()

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

    def send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def send_cmd_repeat(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def repeat_cmd_start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.005, target=self.send_cmd_repeat, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def get_observations(self):
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
            self.commands,
            joint_q_isaac,
            joint_dq_isaac,
            self.actions,
        ))

        # Prepare observations for the policy
        self.env.update_obs(obs)

    def rl_policy_cmd(self):
        # Run inference
        with torch.inference_mode():
            actions, _ = self.policy(self.env.obs, self.env.history_obs)
            self.actions = actions.squeeze(0).detach().cpu().numpy()

        # Map action to joint position targets
        joint_positions = self.actions * self.cfg.action_scale

        # Convert to real joint order and offset by default joint positions
        self.joint_position_targets = np.array([joint_positions[i] for i in self.joint_cmd_order_map], dtype=np.float32)
        self.joint_position_targets += self.default_joint_positions
        self.cmd_lock.acquire()
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.joint_position_targets[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 20.0
            self.low_cmd.motor_cmd[i].kd = 0.5
            self.low_cmd.motor_cmd[i].tau = 0
        self.cmd_lock.release()
        # for _ in range(4):
        #     self.send_cmd()
        #     time.sleep(self.dt / 4.0)
        self.send_cmd()

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
    
    def on_press(self, key):
        try:
            if key.char == keyboard.Key.esc:
                print(' key received, exit all')
                self.terminate = True
            elif key.char == 'q':
                print(' key received, start standing mode')
                self.standing = True
                self.damping = False
                self.rl_policy = False
            elif key.char == 'w':
                print(' key received, start RL policy mode')
                self.standing = False
                self.damping = False
                self.rl_policy = True
            elif key.char == 'e':
                print(' key received, start damping mode')
                self.standing = False
                self.damping = True
                self.rl_policy = False
            elif key.char == '8':
                self.commands[0] = self.commands[0] + 0.1
            elif key.char == '2':
                self.commands[0] = self.commands[0] - 0.1
            elif key.char == '7':
                self.commands[2] = self.commands[2] + 0.1
            elif key.char == '9':
                self.commands[2] = self.commands[2] - 0.1
            elif key.char == '1':
                self.commands[2] = -1.0
            elif key.char == '3':
                self.commands[2] = 1.0
            elif key.char == '6':
                self.commands[1] = self.commands[1] + 0.1
            elif key.char == '4':
                self.commands[1] = self.commands[1] - 0.1
            elif key.char == '0':
                self.commands[0] = 0.0
                self.commands[1] = 0.0
                self.commands[2] = 0.0

            self.commands[0] = min(max(self.commands[0], -self.cfg.max_vel_x), self.cfg.max_vel_x)
            self.commands[1] = min(max(self.commands[1], -self.cfg.max_vel_y), self.cfg.max_vel_y)
            self.commands[2] = min(max(self.commands[2], -self.cfg.max_vel_z), self.cfg.max_vel_z)
            print('user commands: ', self.commands)

        except AttributeError:
            print('keyboard error or no key pressed')

    def print_info(self):
        # print(f"\nJoint position: \nFR {self.proprioceptive_state[7:10].tolist()} \nFL {self.proprioceptive_state[10:13].tolist()} "
        #       f"\nBR {self.proprioceptive_state[13:16].tolist()} \nBL {self.proprioceptive_state[16:19].tolist()}")
        print('user command: ', self.commands)
        print(f"Joint position targets: \nFR {self.joint_position_targets[:3].tolist()} \nFL {self.joint_position_targets[3:6].tolist()} "
              f"\nBR {self.joint_position_targets[6:9].tolist()} \nBL {self.joint_position_targets[9:].tolist()}")

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enx9c69d342225b", help="network interface")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)

    control_dt = 0.02

    custom = Custom(control_dt)
    custom.Init()
    custom.zero_torque_state()
    custom.move_to_default_pos()
    custom.keep_default_pos_state()
    # custom.repeat_cmd_start()
    
    count = 0
    while True:
        try:
            if custom.terminate:
                break

            t_start = time.time()

            if custom.rl_policy:
                custom.rl_policy_cmd()

            if custom.damping:
                custom.damping_cmd()

            # if count % 20 == 0:
            #     custom.print_info()

            t_end = time.time()
            elapsed_dt = t_end - t_start
            if elapsed_dt < control_dt:
                time.sleep(control_dt - elapsed_dt)

            if custom.rl_policy:
                custom.get_observations()
            count = count + 1
        except KeyboardInterrupt:
            break
    print("Enter damping control")
    custom.damping_cmd()
    print('Exit')