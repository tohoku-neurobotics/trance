from isaaclab.utils import configclass

from .env_cfg.observation_cfg import TeacherObsCfg, StudentObsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from .env_cfg import (
    Go2CommandsCfg,
    Go2TerminationsCfg,
    Go2RewardsCfg,
    Go2ActionCfg,
    Go2EventCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class TeacherObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    policy: TeacherObsCfg = TeacherObsCfg()


@configclass
class StudentObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()


@configclass
class Go2TeacherRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Go2RewardsCfg = Go2RewardsCfg()
    terminations: Go2TerminationsCfg = Go2TerminationsCfg()
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    events: Go2EventCfg = Go2EventCfg()
    commands: Go2CommandsCfg = Go2CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


@configclass
class Go2TeacherFlatEnvCfg(Go2TeacherRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore


@configclass
class Go2StudentRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Go2RewardsCfg = Go2RewardsCfg()
    terminations: Go2TerminationsCfg = Go2TerminationsCfg()
    actions: Go2ActionCfg = Go2ActionCfg()
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    events: Go2EventCfg = Go2EventCfg()
    commands: Go2CommandsCfg = Go2CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22

        # Scene
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


@configclass
class Go2StudentFlatEnvCfg(Go2StudentRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.teacher.height_scan = None  # type: ignore
