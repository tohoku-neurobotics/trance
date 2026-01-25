# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from modules.cpg_cfg import CPG, CPGCfg
from modules.reflex import Reflex, ReflexCfg

class L2TCPGVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use l2t actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "teacher" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None, clip_observations: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions
        self.clip_observations = clip_observations

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["student"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["student"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "teacher" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["teacher"][0]
        elif hasattr(self.unwrapped, "num_states") and "teacher" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["teacher"])
        else:
            self.num_privileged_obs = 0

        # cpg module
        self.cpg_cfg = CPGCfg()
        self.cpg = CPG(
            device=self.device,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            decimation=self.cfg.decimation,
            config=self.cpg_cfg,
        )

        self.cpg_num_actions = self.num_actions + self.cfg.reflex_action_num # for joint correction
        self.reflex_cfg = ReflexCfg()
        self.reflex_cfg.skip_network = False
        self.reflex = Reflex(
            device=self.device,
            num_envs=self.num_envs,
            dt=self.cfg.sim.dt,
            decimation=self.cfg.decimation,
            config=self.reflex_cfg,
        )
        
        default_joint_pos_tmp = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  
                                              1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000], 
                                              dtype=torch.float, device=self.device)
        self.default_joint_pos = default_joint_pos_tmp.unsqueeze(0).expand(self.num_envs, -1) 

        joint_limits_tmp = torch.tensor([[-1.0472,  1.0472],
                                         [-1.0472,  1.0472],
                                         [-1.0472,  1.0472],
                                         [-1.0472,  1.0472],
                                         [-1.5708,  3.4907],
                                         [-1.5708,  3.4907],
                                         [-0.5236,  4.5379],
                                         [-0.5236,  4.5379],
                                         [-2.7227, -0.8378],
                                         [-2.7227, -0.8378],
                                         [-2.7227, -0.8378],
                                         [-2.7227, -0.8378]], dtype=torch.float, device=self.device)
        self.joint_limits = joint_limits_tmp.unsqueeze(0).expand(self.num_envs, 12, 2)

        # rx, ry, theta, rx_dot, ry_dot, omega (6 * 4 feet), 2 foot swing parameters, and 1 body height
        self.cpg_num_obs = self.num_obs + self.cpg_cfg.cpg_state_num
        self.cpg_num_privileged_obs = self.num_privileged_obs + self.cpg_cfg.cpg_state_num

        self.cpg_obs = torch.zeros(self.num_envs, self.cpg_num_obs, device=self.device, dtype=torch.float)
        self.cpg_privileged_obs = torch.zeros(self.num_envs, self.cpg_num_privileged_obs, device=self.device, dtype=torch.float)

        # modify the action space to the clip range
        self._modify_action_space()
        self._modify_observation_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.unwrapped.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.unwrapped.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> dict:
        obs_all = {
            "student": self.cpg_obs,
            "teacher": self.cpg_privileged_obs,
        }
        # return observations
        return obs_all

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> dict:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.cpg.reset(env_ids)
        self.reflex.reset(env_ids)
        obs = obs_dict["student"]
        privileged_obs = obs_dict["teacher"]
        cpg_states = self.cpg.get_cpg_states(normalize=True)
        cpg_params = self.cpg.get_cpg_design_params()
        self.cpg_obs = torch.cat((obs, cpg_states, cpg_params), dim=-1)
        self.cpg_privileged_obs = torch.cat((privileged_obs, cpg_states, cpg_params), dim=-1)
        obs_all = {
            "student": self.cpg_obs,
            "teacher": self.cpg_privileged_obs,
        }
        # return observations
        return obs_all

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        actions_tmp = actions.clone()
        if self.clip_actions is not None:
            actions_tmp = torch.clamp(actions_tmp, -self.clip_actions, self.clip_actions)

        cpg_actions = actions_tmp[:, :12]
        reflex_actions = actions_tmp[:, 12:]
        
        processed_actions = self.reflex.get_adjustments(reflex_actions * self.cfg.actions.joint_pos.scale)
        processed_actions = processed_actions.permute(0, 2, 1).reshape(-1, 12)

        foot_position_targets = self.cpg.map_foot_positions(cpg_actions * self.cfg.actions.joint_pos.scale)
        joint_position_targets = self.cpg.compute_inverse_kinematics(foot_position_targets)
        joint_position_targets = joint_position_targets.permute(0, 2, 1).reshape(-1, 12)
        joint_position_targets = processed_actions + joint_position_targets
        joint_position_targets = torch.clip(
            joint_position_targets.clone(),
            self.joint_limits[..., 0],
            self.joint_limits[..., 1],
        )
        processed_actions = (joint_position_targets - self.default_joint_pos) / self.cfg.actions.joint_pos.scale

        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(processed_actions)
        reset_env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.cpg.reset(reset_env_ids)
            self.reflex.reset(reset_env_ids)
        obs = obs_dict["student"]
        privileged_obs = obs_dict['teacher']
        cpg_states = self.cpg.get_cpg_states(normalize=True)
        cpg_params = self.cpg.get_cpg_design_params()
        self.cpg_obs = torch.cat((obs, cpg_states, cpg_params), dim=-1)
        self.cpg_privileged_obs = torch.cat((privileged_obs, cpg_states, cpg_params), dim=-1)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs_all = {
            "student": self.cpg_obs,
            "teacher": self.cpg_privileged_obs
        }
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs_all, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        
        self.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.cpg_num_actions,)
        )
        self.unwrapped.action_space = self.unwrapped.single_action_space

    def _modify_observation_space(self):
        if self.clip_observations is None:
            return
        
        single_teacher_observation_space = gym.spaces.Box(
            low=-self.clip_observations, high=self.clip_observations, shape=(self.cpg_num_privileged_obs,)
        )
        single_student_observation_space = gym.spaces.Box(
            low=-self.clip_observations, high=self.clip_observations, shape=(self.cpg_num_obs,)
        )
        self.unwrapped.single_observation_space = gym.spaces.Dict({
            'teacher': single_teacher_observation_space,
            'student': single_student_observation_space
        })
        self.unwrapped.observation_space = self.unwrapped.single_observation_space