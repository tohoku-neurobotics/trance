# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg

from .locomotion_velocity_rough_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

EVAL_DISCRETE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(60.0, 60.0),
    border_width=0.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.01, 0.06), platform_width=1.0
        ),
    },
)

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    history_len : int = 10
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"

        # rewards, the following reward setting results in worse learning
        # please only consider it as an ablation
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # only trained for the unified policy


        ### original reward
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.05
        # self.rewards.flat_orientation_l2.weight = -0.2
        # self.rewards.base_height_l2.weight = -1.0
        # self.rewards.feet_air_time.weight = 0.1
        # self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.joint_powers.weight = -2e-5

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class NavUnitreeGo2RoughEnvCfg(UnitreeGo2RoughEnvCfg):
    viewer = ViewerCfg(eye=(2.0, 2.0, 2.0), 
                       lookat=(0.0, 0.0, 0.0), 
                       origin_type="asset_body", 
                       env_index=0, 
                       asset_name="robot", 
                       body_name="base")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.time_out = None  # no time_out for navigation task

        # make a smaller scene for play
        self.scene.num_envs = 1  # must be 1 for navigation training
        self.scene.terrain.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}

@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

        # self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        # self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
 
@configclass
class UnitreeGo2RoughEnvCfg_EVAL(UnitreeGo2RoughEnvCfg):
    viewer = ViewerCfg(eye=(2.0, 2.0, 2.0), 
                       lookat=(0.0, 0.0, 0.0), 
                       origin_type="asset_body", 
                       env_index=0, 
                       asset_name="robot", 
                       body_name="base")
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.episode_length_s = 600
        self.scene.terrain.terrain_generator = EVAL_DISCRETE_TERRAINS_CFG

        # disable randomization for play
        self.observations.policy.enable_corruption = False
    
