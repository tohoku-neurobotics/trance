# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from Stable-Baselines3."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=512, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Go2-Play-L2T-v0", help="Name of the task.")
parser.add_argument(
    "--checkpoint", type=str, default="2025-06-02_19-18-45", help="Path to model checkpoint."
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video while playing the checkpoint.",
)

parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")

parser.add_argument(
    "--l2t",
    action="store_true",
    default=True,
    help="Use L2T model for playing.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime

from sb3 import process_sb3_cfg, L2tSb3VecEnvGPUWrapper
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.dict import print_dict

import locomotion.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from rlopt.agent import RecurrentL2T


def main_recurrentl2t_student():

    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore

    # directory for logging into
    log_dir = os.path.join(
        "logs", "l2t", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % 1 == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    # wrap around environment for stable baselines
    env = L2tSb3VecEnvGPUWrapper(env)  # type: ignore

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg
            and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "l2t", "Isaac-Velocity-Rough-Go2-L2T-v0")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    load_run = "2025-08-21_16-54-00"
    load_checkpoint = "model_475136000_steps.zip"
    resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    agent = RecurrentL2T.load(resume_path, env, print_system_info=True, device="cuda:0", custom_objects={"_last_obs": None})
    _last_episode_starts = None

    _last_lstm_states = None
    episode_starts = None
    # reset environment
    obs = env.reset()
    d_vx_data = []
    d_vy_data = []
    d_wz_data = []
    timestep = 0
    i = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            obs = {"student": obs["student"].cpu().numpy()}

            # agent stepping
            # actions, _ = agent.student_predict(obs, deterministic=True)  # type: ignore
            actions, lstm_states = agent.student_policy.predict(
                obs["student"],
                _last_lstm_states,
                episode_starts,
            )
            # env stepping
            obs, reward, dones, info = env.step(actions)

            _last_lstm_states = lstm_states
            _last_episode_starts = dones
            episode_starts = (
                _last_episode_starts.clone().to(agent.device).type(torch.float32)
            )
        privileged_obs_numpy = obs["teacher"].detach().clone().cpu().numpy()
        d_vx = np.fabs(privileged_obs_numpy[:, 0] - privileged_obs_numpy[:, 9])
        d_vy = np.fabs(privileged_obs_numpy[:, 1] - privileged_obs_numpy[:, 10])
        d_wz = np.fabs(privileged_obs_numpy[:, 5] - privileged_obs_numpy[:, 11])
        d_vx_data.append(d_vx)
        d_vy_data.append(d_vy)
        d_wz_data.append(d_wz)
  
        timestep += 1
        # Exit the play loop after recording one video
        if timestep == args_cli.video_length:
            break
    tracking_error_data = {'d_vx': np.array(d_vx_data),
                           'd_vy': np.array(d_vy_data),
                           'd_wz': np.array(d_wz_data)}
    np.savez_compressed(os.path.join(log_root_path, load_run, "tracking_error_data.npz"), **tracking_error_data)
    # close the simulator
    env.close()


if __name__ == "__main__":
    if args_cli.l2t:
        # run the main function
        # main_l2t_student()
        # main_recurrentl2t_teacher()
        main_recurrentl2t_student()
    else:
        # run the main function
        main()
    # close sim app
    simulation_app.close()