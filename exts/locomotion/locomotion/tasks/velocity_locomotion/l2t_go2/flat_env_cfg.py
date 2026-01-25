# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import Go2RoughEnvCfg


@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        # self.observations.policy.enable_corruption = False
        # self.observations.student.enable_corruption = (
        #     False  # remove random pushing event
        # )
        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.teacher.height_scan = None  # type: ignore
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
