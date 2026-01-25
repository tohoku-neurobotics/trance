import gymnasium as gym
import os
import copy
import torch

from cpg.tasks.cpg.dream_waq_agent.on_policy_runner_v2 import OnPolicyRunner

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils import parse_env_cfg
from cpg.tasks.cpg.agents.dream_waq_ppo_cfg import Go2RoughPPOCfg
from cpg.tasks.cpg.dream_waq_env.dream_waq_go2_rough_env_cfg import NavUnitreeGo2RoughEnvCfg
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
    print("latent mu model", model)
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path2)

    path3 = os.path.join(path, 'latent_var_dwaq.pt')
    model = copy.deepcopy(actor_critic.encode_logvar_latent).to('cpu')
    print("latent var model", model)
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


def load_policy():
    env_cfg: NavUnitreeGo2RoughEnvCfg = NavUnitreeGo2RoughEnvCfg()
    agent_cfg: Go2RoughPPOCfg = Go2RoughPPOCfg()
    
    video = False
    video_interval = 1000
    video_length = 1000

    # specify directory for logging experiments
    log_root_path = os.path.join("../logs", "dream_waq_v2", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    agent_cfg.load_run = "2025-12-05_15-01-56_dream_waq_no_y"
    agent_cfg.load_checkpoint = "model_5800.pt"
    # agent_cfg.load_run = "2025-09-27_21-30-17_dream_waq_cnn_decoder_more_random"
    # agent_cfg.load_checkpoint = "model_2200.pt"
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make("Dream-WaQ-Rough-Unitree-Go2-Nav-v2", cfg=env_cfg, render_mode="rgb_array" if video else None)
    # wrap for video recording
    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "nav"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = DreamWaQVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # create runner from dream-waq
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy_encoder(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_policy_as_jit_actor(runner.alg.actor_critic, export_model_dir)
    export_policy_as_jit_encoder(runner.alg.actor_critic, export_model_dir)
    
    return policy, env



