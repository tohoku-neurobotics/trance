from isaaclab.utils import configclass

from .locomotion_velocity_rough_env_cfg import LocomotionVelocityRoughEnvCfg
from .observation_cfg import TeacherObsCfg, StudentObsCfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class L2TObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()

# default env
@configclass
class Go2L2TRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: L2TObservationsCfg = L2TObservationsCfg()

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

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class Go2L2TRoughEnvCfg_PLAY(Go2L2TRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


        # disable randomization for play
        self.observations.teacher.enable_corruption = False
        self.observations.student.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2L2TFlatEnvCfg(Go2L2TRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.teacher.height_scan = None  # type: ignore


class Go2L2TFlatEnvCfg_PLAY(Go2L2TFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.teacher.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
