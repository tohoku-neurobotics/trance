from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import ContactSensor

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import torch
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject

from isaaclab.envs import ManagerBasedRLEnv

def get_contact_states(env: ManagerBasedRLEnv,
                       sensor_cfg : SceneEntityCfg,
                       range_normalization : tuple[float, float]
                       ) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_scale = 2. / (range_normalization[1] - range_normalization[0])
    contact_force_shift = (range_normalization[1] + range_normalization[0]) / 2.
    # 0 means most recent contact state

    base_id, _ = contact_sensor.find_bodies("base") 
    foot_ids, _ = contact_sensor.find_bodies(".*_foot")
    calf_ids, _ = contact_sensor.find_bodies(".*_calf")
    thigh_ids, _ = contact_sensor.find_bodies(".*_thigh")
    hip_ids, _ = contact_sensor.find_bodies(".*_hip")
    contact_states_ids = base_id + hip_ids + thigh_ids + calf_ids + foot_ids

    contact_states = contact_sensor.data.net_forces_w_history[:, :, contact_states_ids, 0]
    contact_states_flatten = contact_states.clone().view(env.num_envs, -1)
    return (contact_states_flatten - contact_force_shift) * contact_force_scale

@configclass
class TeacherObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)
    contact_states = ObsTerm(
        func=get_contact_states,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"),
                "range_normalization": (0.0, 50.0)})

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class StudentObsCfg(ObsGroup):
    """Observations for student group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
