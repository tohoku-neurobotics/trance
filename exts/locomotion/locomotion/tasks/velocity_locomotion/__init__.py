# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Anymal-C locomotion environment.
"""

import gymnasium as gym

from . import agents

from .l2t_go2 import go2_agents, l2t_env_cfg
from .dream_waq_env import dream_waq_go2_rough_env_cfg, dream_waq_nav_env, dream_waq_a1_rough_env_cfg
from .slr_rl_env import slr_go2_rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="SLR-Rough-Unitree-Go2-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": slr_go2_rough_env_cfg.UnitreeGo2RoughEnvCfg,
        "agent_cfg_entry_point": f"{agents.__name__}.slr_ppo_cfg:Go2RoughPPOCfg",
    },
)

gym.register(
    id="Dream-WaQ-Rough-Unitree-Go2-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dream_waq_go2_rough_env_cfg.UnitreeGo2RoughEnvCfg,
        "agent_cfg_entry_point": f"{agents.__name__}.dream_waq_ppo_cfg:Go2RoughPPOCfg",
    },
)

gym.register(
    id="Dream-WaQ-Rough-Unitree-A1-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dream_waq_a1_rough_env_cfg.UnitreeA1RoughEnvCfg,
        "agent_cfg_entry_point": f"{agents.__name__}.dream_waq_ppo_cfg:A1RoughPPOCfg",
    },
)

gym.register(
    id="Dream-WaQ-Rough-Unitree-A1-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dream_waq_a1_rough_env_cfg.UnitreeA1RoughEnvCfg_PLAY,
        "agent_cfg_entry_point": f"{agents.__name__}.dream_waq_ppo_cfg:A1RoughPPOCfg",
    },
)

gym.register(
    id="Dream-WaQ-Rough-Unitree-Go2-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dream_waq_go2_rough_env_cfg.UnitreeGo2RoughEnvCfg_PLAY,
        "agent_cfg_entry_point": f"{agents.__name__}.dream_waq_ppo_cfg:Go2RoughPPOCfg",
    },
)

gym.register(
    id="Dream-WaQ-Rough-Unitree-Go2-Eval-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dream_waq_go2_rough_env_cfg.UnitreeGo2RoughEnvCfg_EVAL,
        "agent_cfg_entry_point": f"{agents.__name__}.dream_waq_ppo_cfg:Go2RoughPPOCfg",
    },
)

gym.register(
    id="Dream-WaQ-Rough-Unitree-Go2-Nav-v2",
    entry_point=dream_waq_nav_env.UnitreeGo2NavEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dream_waq_go2_rough_env_cfg.NavUnitreeGo2RoughEnvCfg,
        "agent_cfg_entry_point": f"{agents.__name__}.dream_waq_ppo_cfg:Go2RoughPPOCfg",
    },
)

# for l2t
gym.register(
    id="Isaac-Velocity-Rough-Go2-L2T-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": l2t_env_cfg.Go2L2TRoughEnvCfg,
        "rsl_rl_cfg_entry_point": go2_agents.rsl_rl_cfg.Go2RoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{go2_agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{go2_agents.__name__}:sb3_l2t_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Play-L2T-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": l2t_env_cfg.Go2L2TRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": go2_agents.rsl_rl_cfg.Go2RoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{go2_agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{go2_agents.__name__}:sb3_l2t_cfg.yaml",
    },
)


