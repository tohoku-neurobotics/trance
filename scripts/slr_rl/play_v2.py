# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with dream-waq."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import copy
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with dream-waq.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="SLR-Rough-Unitree-Go2-v2", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from locomotion.tasks.velocity_locomotion.agents.actor_critic_slr import ActorCriticSLR
from locomotion.tasks.velocity_locomotion.slr_rl_env.slr_env_wrapped import SLRVecEnvWrapper

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from locomotion.tasks.velocity_locomotion.agents.slr_ppo_cfg import Go2RoughPPOCfg

# Import extensions to set up environment tasks
import locomotion.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def export_policy_as_jit_actor(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'actor_dwaq.pt')
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    print("policy model", model)        
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)

def export_policy_as_jit_encoder(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path1 = os.path.join(path, 'encoder_dwaq.pt')
    model = copy.deepcopy(actor_critic.encoder).to('cpu')
    print("encoder model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path1)

    path2 = os.path.join(path, 'latent_mu_dwaq.pt')
    model = copy.deepcopy(actor_critic.encode_mean_latent).to('cpu')
    print("latent mu model",model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path2)

    path3 = os.path.join(path, 'latent_var_dwaq.pt')
    model = copy.deepcopy(actor_critic.encode_logvar_latent).to('cpu')
    print("latent var model",model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path3)

    path4 = os.path.join(path, 'vel_mu_dwaq.pt')
    model = copy.deepcopy(actor_critic.encode_mean_vel).to('cpu')
    print("vel mu model",model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path4)

    path5 = os.path.join(path, 'vel_var_dwaq.pt')
    model = copy.deepcopy(actor_critic.encode_logvar_vel).to('cpu')
    print("vel var model",model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path5)

@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: Go2RoughPPOCfg):
    """Train with dream-waq agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "slr_rl_v2", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    agent_cfg.load_run = "2025-06-08_17-59-41_slr"
    agent_cfg.load_checkpoint = "model_1600.pt" # 1700 nothing shows up in isaacsim
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SLRVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    actor_critic_class = eval(agent_cfg.policy_class_name)
    policy: ActorCriticSLR  = actor_critic_class(
                                            num_props=env.num_obs,
                                            num_hist=env.history_len,
                                            num_actions=env.num_actions,
                                            )
    print(policy)
    model_dict = torch.load(resume_path, map_location='cuda:0')
    policy.load_state_dict(model_dict['model_state_dict'])
    policy.eval()

    policy.save_torch_jit_policy()
    policy.to(env.device)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy.act_inference(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
         
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
