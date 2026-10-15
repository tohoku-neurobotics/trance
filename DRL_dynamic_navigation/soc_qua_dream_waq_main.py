import argparse
import numpy as np
import os
import torch

from algos.SAC.sac import SacAeAgent
from algos.SAC.utils import ReplayBuffer, eval_policy_mode, set_seed_everywhere
from info import *
from env_soc_qua_dream_waq import CrowdSim
from torch.utils.tensorboard import SummaryWriter

import argparse
from isaaclab.app import AppLauncher

    
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
    specified_seed = [12, 87, 103 , 458, 474,
                      48, 82, 211, 226, 295, 324, 368, 398, 480,
                      98, 102, 180, 222, 261, 329, 359, 371, 426, 453]
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
            # action = eval_env.cal_dwa_action()
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
                    elif isinstance(info, Fall):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_env.global_time)
                        print('evaluation episode ' + str(i) + ', fall occur at evaluation step: ' + str(ep_step))      
                    elif isinstance(info, OutArea):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_env.global_time)
                        print('evaluation episode ' + str(i) + ', outarea occur at evaluation step: ' + str(ep_step))         
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        if save_directory is not None:
            if isinstance(info, ReachGoal):
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '.npz'
            elif isinstance(info, Fall):
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fall' + '.npz'
            else:
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
    
            if if_save_data:
                np.savez_compressed(file_name, **eval_env.log_env)

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
    # args.load_model, format, step_NO
    parser.add_argument("--load_model", type=str, default="")

    # step_940000
    parser.add_argument("--load_test_model", type=str, default="step_1560000")

    parser.add_argument("--init_test", action="store_true", default=False)

    parser.add_argument("--warm_start", action="store_true", default=True)
    parser.add_argument("--clip_action", action="store_true", default=True)
    parser.add_argument("--penalty_rotation", action="store_true", default=True)
    # environment settings
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument('--frame_stack', default=6, type=int)
    parser.add_argument("--random_human_v_pref", action="store_true", default=False)
    parser.add_argument("--action_penalty", action="store_true", default=False)
    # 2 robot speed, 2 local goal
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    # other option: terrain_dim can be obs + encoder dim from cenet (45 + 19)
    parser.add_argument("--terrain_dim", type=int, default=64)

    # for quadruped
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=800, help="Length of the recorded video (in steps).")
    parser.add_argument("--task", type=str, default="Dream-WaQ-Rough-Unitree-Go2-Nav-v2", help="Name of the task.")

    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()

    device = torch.device(args.device)

    # always enable cameras to record video
    if args.video:
        args.enable_cameras = True
    if args.robot_model == 'quadruped':
        # launch omniverse app
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        from load_quadruped_policy import load_policy

        quadruped_policy, quadruped_env = load_policy()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, Seed: {args.seed}")
    print("---------------------------------------")
    if args.robot_model == "quadruped":
        file_prefix = './logs/' + args.policy + '_' + args.robot_model + '_' + args.lidar_obs_type + '_dream_waq_cnn'
    else:
        file_prefix = './logs/' + args.policy + '_' + args.robot_model + '_' + args.lidar_obs_type # + '_backward'  # response to tase
    if args.action_penalty:
        file_prefix = file_prefix + '_action_penalty'
    if args.random_human_v_pref:
        file_prefix = file_prefix + '_random_human'
    if args.penalty_rotation:
        file_prefix = file_prefix + '_penalty_rotation'
    if args.robot_model == "quadruped":
        file_prefix = file_prefix + '_' + str(args.terrain_dim)
    if not args.warm_start:
        file_prefix = file_prefix + '/seed_' + str(args.seed) + '_no_warm' 
    else:
        file_prefix = file_prefix + '/seed_' + str(args.seed)
    if not args.clip_action:
        file_prefix = file_prefix + '_no_clip' 
    else:
        file_prefix = file_prefix + '_clip_in_place_rotation' # new
  
    file_results = file_prefix + '/results'
    file_models = file_prefix + '/models'
    file_evaluation_episodes = file_prefix + '/evaluation_episodes'
    if args.load_test_model != '':
        file_final_test_episodes = file_prefix + '/final_test_episodes' + '_' + args.load_test_model
    else:
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
                             [0.5,  1.0]])  # clip_in_place_rotation

    # action_range = np.array([[-0.5, -1.0],
    #                          [0.5,  1.0]])  # backward
    
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
        terrain_dim = args.terrain_dim

        env = CrowdSim(args, terrain_dim, action_range, device, 
                       quadruped_policy=quadruped_policy, quadruped_env=quadruped_env)
        eval_env = CrowdSim(args, terrain_dim, action_range, device, action_choices=action_choices,
                            quadruped_policy=quadruped_policy, quadruped_env=quadruped_env)
    else:
        terrain_dim = 0
        env = CrowdSim(args, terrain_dim, action_range, device)
        eval_env = CrowdSim(args, terrain_dim, action_range, device)
        
    set_seed_everywhere(args.seed)

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
        agent.load(file_models + '/' + args.load_model, device=device)
        # args.load_model, format, step_NO_success_NO
        # extract the first number
        checkpoint_steps = int(args.load_model.split('_')[1])

    # please train the social navigation with differential model at first for curriculum learning
    if args.robot_model == 'quadruped' and args.load_test_model == "" and args.load_model == "": 
        if args.warm_start:
            terrain_dim_old_agent = 0
            old_agent = SacAeAgent(
                obs_shape,
                terrain_dim_old_agent,
                robot_goal_state_dim,
                args.action_dim,
                action_range,
                device
            )
            # old_agent.load('logs/sac_differential_transform/seed_1_frame_6/models/step_3280000_success_98', device=device)
            old_agent.load('logs/sac_differential_transform/seed_1_frame_6/models/step_3680000_success_99', device=device)
            # old_agent.load('logs/sac_differential_transform_backward/seed_1/models/step_4560000_success_98', device=device)
            agent.load("", old_agent=old_agent, device=device)

            if args.init_test:
                print('start initial test')
                test_times = 500
                current_step = 0
                file_final_test_episodes_init = file_final_test_episodes + '_init'
                if not os.path.exists(file_final_test_episodes_init):
                    os.makedirs(file_final_test_episodes_init)
                success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, current_step, 
                                                                        eval_episodes=test_times, 
                                                                        save_directory=file_final_test_episodes_init, 
                                                                        final_test=True)
        
                print('success_rate, collision_rate, avg_nav_time')
                print(success_rate, collision_rate, avg_nav_time)
                return

    if args.load_test_model != "":
        print('start to test')
        agent.load(file_models + '/' + args.load_test_model, device=device)
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
                elif isinstance(info, OutArea):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num + 1) + 
                        ', out_area occur at train step: ' + str(episode_timesteps))        
                elif isinstance(info, Fall):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num + 1) + 
                        ', fall occur at train step: ' + str(episode_timesteps))   
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
        quadruped_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
