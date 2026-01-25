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
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Dream-WaQ-Rough-Unitree-Go2-Play-v2", help="Name of the task.")
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

from cpg.tasks.cpg.dream_waq_agent.on_policy_runner_v2 import OnPolicyRunner

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from cpg.tasks.cpg.agents.dream_waq_ppo_cfg import Go2RoughPPOCfg, A1RoughPPOCfg
from cpg.tasks.cpg.dream_waq_env.dream_waq_env_wrapped import DreamWaQVecEnvWrapper

# Import extensions to set up environment tasks
import cpg.tasks  # noqa: F401

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
    print("vel mu model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path4)

    path5 = os.path.join(path, 'vel_var_dwaq.pt')
    model = copy.deepcopy(actor_critic.encode_logvar_vel).to('cpu')
    print("vel var model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path5)


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: Go2RoughPPOCfg | A1RoughPPOCfg):
    """Train with dream-waq agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "dream_waq_v2", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # agent_cfg.load_run = "2025-07-31_14-51-16_dream_waq_cnn_decoder"  # Go2 2500
    # more random
    agent_cfg.load_run = "2025-12-05_15-01-56_dream_waq"   # Go2 more random, 2200
    # agent_cfg.load_run = "2025-07-31_08-08-33_dream_waq_original"   # Go2 original 2400
    # agent_cfg.load_run = "2025-07-31_07-19-04_dream_waq_mlp"   # Go2 mlp 2600
    agent_cfg.load_checkpoint = "model_5800.pt"
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = DreamWaQVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # create runner from dream-waq
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_policy_as_jit_actor(runner.alg.actor_critic, export_model_dir)
    export_policy_as_jit_encoder(runner.alg.actor_critic, export_model_dir)
    
    # reset environment
    obs_dict = env.get_observations()
    obs, obs_hist = obs_dict['obs'], obs_dict['history_obs']
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs.detach(), obs_hist.detach())
            # env stepping
            obs_dict, rewards, dones, infos = env.step(actions)
            obs, privileged_obs, prev_privileged_obs, obs_hist = \
                obs_dict['obs'], obs_dict['privileged_obs'], obs_dict['prev_privileged_obs'], obs_dict['history_obs']
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
