import argparse
import numpy as np
import os
import torch

from algos.SAC.sac import SacAeAgent
from algos.SAC.utils import ReplayBuffer, eval_policy_mode, set_seed_everywhere
from info import *
from env_soc_qua import CrowdSim
from torch.utils.tensorboard import SummaryWriter

import argparse
from isaaclab.app import AppLauncher
# local imports
from scripts import cli_args  # isort: skip

    
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, current_steps, eval_episodes=100, save_directory=None, final_test=False):
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    if_save_data = False
    for i in range(eval_episodes):
        if_save_data = (i < 10 or final_test)
        lidar_image, terrain, robot_goal_state = eval_env.reset(test=final_test, save_data=if_save_data, seed=i)
        # eval_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_env.max_episode_step:
            # t1 = time.time()
            with eval_policy_mode(policy):
                action = policy.select_action(lidar_image, terrain, robot_goal_state)
            lidar_image, terrain, robot_goal_state, reward, done, info = eval_env.step(action, eval=True, save_data=if_save_data)
          
            # print('time: ', time.time() - t1) # 7.5ms
            # eval_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            ep_step = ep_step + 1
            if done or ep_step == eval_env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == eval_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_env.global_time)
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_env.global_time)
                        print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))           
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        if save_directory is not None:
            if isinstance(info, ReachGoal):
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '.npz'
            else:
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
            if if_save_data:
                np.savez_compressed(file_name, **eval_env.log_env)
                # policy.save_features(save_directory)

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_env.time_limit
    
    return success_rate, collision_rate, avg_nav_time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", default="sac")
    # options, differential, quadruped
    parser.add_argument("--robot_model", default="quadruped")
    # option, transform, direct
    parser.add_argument("--lidar_obs_type", default="transform")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=10000, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=20000, type=int)
    # How often (time steps) we save the trained model
    parser.add_argument("--save_model_freq", default=200000, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=6e6, type=int)
    
    # Model load file name, "" doesn't load, "default" uses file_name
    # args.load_model, format, step_NO_success_NO
    # parser.add_argument("--load_model", type=str, default="step_1700000_success_94")
    parser.add_argument("--load_model", type=str, default="")

    parser.add_argument("--load_test_model", type=str, default="")
    # environment settings
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument('--frame_stack', default=6, type=int)
    parser.add_argument("--random_human_v_pref", action="store_true", default=False)
    # 2 robot speed, 2 local goal
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    parser.add_argument("--terrain_enabled", action="store_true", default=True)
    # default terrain height dim, please make sure it is consistent with the data dim of the height sensor
    parser.add_argument("--terrain_dim", type=int, default=70)

    # for quadruped
    # set num_envs as 1 to save model inference time and isaac simulation time
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=800, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--task", type=str, default="CPG-Rough-Unitree-A1-Navigation-v0", help="Name of the task.")

    # for rl-based locomotion and isaac sim
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()

    # always enable cameras to record video
    if args.video:
        args.enable_cameras = True
    if args.robot_model == 'quadruped':
        # launch omniverse app
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab.utils.dict import print_dict
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from isaaclab_rl.rsl_rl import (
            RslRlOnPolicyRunnerCfg,
            RslRlVecEnvWrapper,
            export_policy_as_jit,
            export_policy_as_onnx,
        )
        import cpg.tasks # necessary for gym environments

        """Play with RSL-RL agent."""
        # parse configuration
        quadruped_env_cfg = parse_env_cfg(
            args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args.task, args)

        # specify directory for logging experiments
        log_root_path = os.path.join("../logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        agent_cfg.load_run = "2025-05-14_10-12-57_name"
        agent_cfg.load_checkpoint = "model_2999.pt"
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

        # create isaac environment
        quadruped_envs = gym.make(args.task, cfg=quadruped_env_cfg, render_mode="rgb_array" if args.video else None)
        # wrap for video recording
        if args.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            quadruped_envs = gym.wrappers.RecordVideo(quadruped_envs, **video_kwargs)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(quadruped_envs.unwrapped, DirectMARLEnv):
            quadruped_envs = multi_agent_to_single_agent(quadruped_envs)

        # wrap around environment for rsl-rl
        quadruped_envs = RslRlVecEnvWrapper(quadruped_envs)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(quadruped_envs, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        quadruped_policy = ppo_runner.get_inference_policy(device=quadruped_envs.unwrapped.device)

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, Seed: {args.seed}")
    print("---------------------------------------")
    file_prefix = './logs/' + args.policy + '_' + args.robot_model + '_' + args.lidar_obs_type
    if args.random_human_v_pref:
        file_prefix = file_prefix + '_random_human'
    if args.terrain_enabled:
        file_prefix = file_prefix + '_terrain_enabled'
   
    file_prefix = file_prefix + '/seed_' + str(args.seed)  
    file_results = file_prefix + '/results'
    file_models = file_prefix + '/models'
    file_evaluation_episodes = file_prefix + '/evaluation_episodes'
    file_final_test_episodes = file_prefix + '/final_test_episodes'

    if not os.path.exists(file_results):
        os.makedirs(file_results)

    if not os.path.exists(file_models):
        os.makedirs(file_models)

    if not os.path.exists(file_evaluation_episodes):
        os.makedirs(file_evaluation_episodes)
        
    if not os.path.exists(file_final_test_episodes):
        os.makedirs(file_final_test_episodes)

    writer = SummaryWriter(log_dir=file_results)

    action_range = np.array([[0.0, -1.0],
                             [0.5,  1.0]])
    
    action_resolution = [0.05, 0.1]
    action_num = [int((action_range[1, 0] - action_range[0, 0]) / action_resolution[0]) + 1,
                  int((action_range[1, 1] - action_range[0, 1]) / action_resolution[1]) + 1]
    action_choice_1 = np.linspace(action_range[0, 0], action_range[1, 0], num=action_num[0])
    action_choice_2 = np.linspace(action_range[0, 1], action_range[1, 1], num=action_num[1])
    action_choice_dim = action_num[0] * action_num[1]
    action_choices = np.zeros((action_num[0], action_num[1], 2), dtype=np.float32)
    for i in range(action_num[0]):
        for j in range(action_num[1]):
            action_choices[i, j, :] = np.array([action_choice_1[i], action_choice_2[j]]) 
    action_choices = np.reshape(action_choices, (action_choice_dim, 2))
    
    if args.robot_model == 'quadruped':
        if args.terrain_enabled:
            terrain_dim = quadruped_envs.unwrapped.height_scanner_dim
        else:
            terrain_dim = 0
        env = CrowdSim(args, terrain_dim, action_range, quadruped_policy=quadruped_policy, quadruped_envs=quadruped_envs)
        eval_env = CrowdSim(args, terrain_dim, action_range, quadruped_policy=quadruped_policy, quadruped_envs=quadruped_envs)
    else:
        if args.terrain_enabled:
            terrain_dim = args.terrain_dim
        else:
            terrain_dim = 0
        env = CrowdSim(args, terrain_dim, action_range)
        eval_env = CrowdSim(args, terrain_dim, action_range)
        
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    obs_shape = (1, args.frame_stack, env.n_laser)
    robot_goal_state_dim = args.robot_goal_state_dim
    agent = SacAeAgent(
            obs_shape,
            terrain_dim,
            robot_goal_state_dim,
            args.action_dim,
            action_range,
            device
        )
    
    replay_buffer = ReplayBuffer(
        obs_shape,
        terrain_dim,
        robot_goal_state_dim, 
        args.action_dim,
        device=device
    )

    checkpoint_steps = 0
    if args.load_model != "":
        # replay_buffer.load(file_buffer)
        agent.load(file_models + '/' + args.load_model)
        # args.load_model, format, step_NO_success_NO
        # extract the first number
        checkpoint_steps = int(args.load_model.split('_')[1])

    # please train the social navigation with differential model at first for curriculum learning
    if args.robot_model == 'quadruped' and args.load_test_model == "" and args.load_model == "": 
        terrain_dim_old_agent = 0
        old_agent = SacAeAgent(
            obs_shape,
            terrain_dim_old_agent,
            robot_goal_state_dim,
            args.action_dim,
            action_range,
            device
        )
        old_agent.load('logs/sac_differential_transform/seed_1/models/step_6000000_success_100')
        agent.load("", old_agent=old_agent)

    if args.load_test_model != "":
        print('start to test')
        agent.load(file_models + '/' + args.load_test_model)
        debug = False
        if debug:
            test_times = 1
            current_step = 9999
        else:
            test_times = 500
            current_step = 0
        success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, current_step, 
                                                                 eval_episodes=test_times, save_directory=file_final_test_episodes, 
                                                                 final_test=True)
    
        print('success_rate, collision_rate, avg_nav_time')
        print(success_rate, collision_rate, avg_nav_time)
        return

    evaluations = []

    lidar_image, terrain, robot_goal_state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(checkpoint_steps + 1, int(args.max_timesteps) + 1):
        if t == args.start_timesteps:
            print('replay buffer has been initialized')
        # Perform action
        # sample action for data collection
        if t < args.start_timesteps and args.robot_model != 'quadruped':
            action = np.random.uniform(action_range[0], action_range[1])
        else:
            with eval_policy_mode(agent):
                action = agent.sample_action(lidar_image, terrain, robot_goal_state)
        next_lidar_image, next_terrain, next_robot_goal_state, reward, done, info = env.step(action)

        episode_timesteps += 1

        if episode_timesteps == env.max_episode_step:
            done_bool = 0.0
        else:
            done_bool = float(done)

        replay_buffer.add(lidar_image, terrain, robot_goal_state, action, reward, 
                          next_lidar_image, next_terrain, next_robot_goal_state, done_bool)
    
        lidar_image = next_lidar_image
        terrain = next_terrain
        robot_goal_state = next_robot_goal_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            num_updates = args.start_timesteps if t == args.start_timesteps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, writer, t)

        if done or episode_timesteps == env.max_episode_step or isinstance(info, ReachGoal):
            if episode_timesteps == env.max_episode_step:
                print('total step ' + str(t) + ', train episode ' + str(episode_num + 1) + ', time out: ' + str(episode_timesteps))
            else:
                if isinstance(info, ReachGoal):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num + 1) + 
                        ', goal reaching at train step: ' + str(episode_timesteps))
                elif isinstance(info, Collision):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num + 1) + 
                        ', collision occur at train step: ' + str(episode_timesteps))       
                else:
                    raise ValueError('Invalid end signal from environment')
            lidar_image, terrain, robot_goal_state = env.reset()
            done = False
            episode_num += 1
            writer.add_scalar('train/episode_reward', episode_reward, episode_num)
            episode_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if t % args.eval_freq == 0:
            if args.robot_model == 'quadruped':
                if t % (int(args.save_model_freq / 10)) == 0:
                    file_name = '/step_' + str(t)
                    agent.save(file_models + file_name)
            else:
                success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, t, save_directory=file_evaluation_episodes)
                file_name = '/step_' + str(t) + '_success_' + str(int(success_rate * 100))
                print('success_rate, collision_rate, avg_nav_time at step ' + str(t))
                print(success_rate, collision_rate, avg_nav_time)
                writer.add_scalar('eval/success_rate', success_rate, t)
                writer.add_scalar('eval/collision_rate', collision_rate, t)
                evaluations.append(success_rate)
                np.savetxt(file_results + file_name + '.txt', evaluations)
                if success_rate > 0.85 or t % args.save_model_freq == 0:
                    agent.save(file_models + file_name)
         
    print('final test')
    success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, t, 
                                                             eval_episodes=500, save_directory=file_final_test_episodes,
                                                             final_test=True)
    print('success_rate, collision_rate, avg_nav_time')
    print(success_rate, collision_rate, avg_nav_time)

    if args.robot_model == 'quadruped':
        simulation_app.close()


if __name__ == "__main__":
    main()
