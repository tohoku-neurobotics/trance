from __future__ import annotations

import gymnasium as gym
import torch


from .dream_waq_go2_rough_env_cfg import NavUnitreeGo2RoughEnvCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

import isaaclab.utils.math as math_utils

import numpy as np

class UnitreeGo2NavEnv(ManagerBasedRLEnv):
    cfg: NavUnitreeGo2RoughEnvCfg
    
    def __init__(
        self,
        cfg: NavUnitreeGo2RoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg = cfg

        default_joint_pos_tmp = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  
                                              1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000], 
                                              dtype=torch.float, device=self.device)
        self.default_joint_pos = default_joint_pos_tmp.unsqueeze(0).expand(self.num_envs, -1) 
        self.default_joint_vel = torch.zeros_like(self.default_joint_pos)
        default_root_state_tmp = torch.tensor([0.0000, 0.0000, 0.4000, 
                                               1.0000, 0.0000, 0.0000, 0.0000, 
                                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 
                                               dtype=torch.float, device=self.device)
        self.default_root_state = default_root_state_tmp.unsqueeze(0).expand(self.num_envs, -1) 

        self.row_terrains = self.scene.terrain.terrain_origins.shape[0]
        self.col_terrains = self.scene.terrain.terrain_origins.shape[1]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset robot state
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = self.default_joint_vel[env_ids]
        default_root_state = self.default_root_state[env_ids]

        # poses
        range_list = [self.cfg.events.reset_base.params['pose_range'].get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        row_terrain_origin_choice = np.random.choice(self.row_terrains)
        col_terrain_origin_choice = np.random.choice(self.col_terrains)
        env_origin = self.scene.terrain.terrain_origins[row_terrain_origin_choice, col_terrain_origin_choice]
 
        positions = default_root_state[:, 0:3] + env_origin + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(default_root_state[:, 3:7], orientations_delta)

        self.scene.articulations['robot'].write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.scene.articulations['robot'].write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.scene.articulations['robot'].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    def get_robot_pose(self, env_id: int):
        # get global xy position and yaw angle
        state = self.scene.get_state()
        root_pose = state['articulation']['robot']['root_pose']
        xy_w = root_pose[env_id, :2]
        quat_w = root_pose[env_id, 3:7] # (w, x, y, z)
        w, x, y, z = quat_w.unbind(-1)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_w = torch.atan2(siny_cosp, cosy_cosp)
        torch_tensor = torch.tensor([xy_w[0], xy_w[1], yaw_w], dtype=torch.float32, device=self.device)
        robot_pose = torch_tensor.cpu().numpy()
        return robot_pose
    




  
