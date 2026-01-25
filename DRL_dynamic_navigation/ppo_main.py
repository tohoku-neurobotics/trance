import argparse
import numpy as np
import random
import os
import torch

from algos.PPO.ppo import PPO
from algos.PPO.utils import set_seed_everywhere
from info import *
from crowd_sim_complex_constraint_sac import CrowdSim
from torch.utils.tensorboard import SummaryWriter

import time


    
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, current_steps, 
                eval_episodes=100, save_directory=None, 
                final_test=False):
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
        lidar_image, robot_goal_state  = eval_env.reset(test=final_test, save_data=if_save_data, seed=i)
        # eval_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_env.max_episode_step:
            # t1 = time.time()
            action = policy.evaluation_action(lidar_image, robot_goal_state)
            # action = eval_env.dwa_compute_action()
            lidar_image, robot_goal_state, reward, done, info = eval_env.step(action, eval=True, save_data=if_save_data)
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

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_env.time_limit
    
    return success_rate, collision_rate, avg_nav_time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", default="ppo")
    # options, differential, quadruped
    parser.add_argument("--robot_model", default="differential")
    # options, differential, quadruped
    parser.add_argument("--robot_test_model", default="differential")
    # option, transform, direct
    parser.add_argument("--lidar_obs_type", default="transform")
    # device
    parser.add_argument("--device", type=str, default='cuda')
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=20000, type=int)
    # How often (time steps) we save the trained model
    parser.add_argument("--save_model_freq", default=200000, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=6e6, type=int)

    # update times
    parser.add_argument("--ppo_epochs", default=10, type=int)
    parser.add_argument("--ppo_update_timestep", default=2000, type=int)
    # action_std decay frequency (in num timesteps)
    parser.add_argument("--action_std_decay_freq", default=250000, type=int)
    parser.add_argument("--action_std_decay_rate", default=0.05, type=float)
    # minimum action_std (stop decay after action_std <= min_action_std)
    parser.add_argument("--min_action_std", default=0.1, type=float)

    # parser.add_argument("--load_test_model", type=str, default="step_1700000_success_94")

    parser.add_argument("--load_test_model", type=str, default="")
    # environment settings
    parser.add_argument("--random_human_v_pref", action="store_true", default=False)
    parser.add_argument('--frame_stack', default=6, type=int)
    # 2 robot speed, 2 local goal
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, Seed: {args.seed}")
    print("---------------------------------------")
    file_prefix = './logs/' + args.policy + '_' + args.robot_model + '_' + args.lidar_obs_type
    if args.random_human_v_pref:
        file_prefix = file_prefix + '_random_human'
   
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
   
    env = CrowdSim(args, action_range)
    eval_env = CrowdSim(args, action_range)

    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    obs_shape = (1, args.frame_stack, env.n_laser)
    agent = PPO(obs_shape, args.robot_goal_state_dim, args.action_dim, action_range, device)

    if args.load_test_model != "":
        print('start to test')
        agent.load(torch.load(file_models + '/' + args.load_test_model + "_ppo.pth"))
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
    lidar_image, robot_goal_state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    traj_length = 0
    for t in range(1, 1 + int(args.max_timesteps)):
        action = agent.select_action(lidar_image, robot_goal_state)
        next_lidar_image, next_robot_goal_state, reward, done, info = env.step(action)
        episode_timesteps = episode_timesteps + 1

        if episode_timesteps == env.max_episode_step:
            dw = True
        else:
            dw = False

        # saving reward and is_terminals
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done or dw)
    
        lidar_image = next_lidar_image
        robot_goal_state = next_robot_goal_state
        episode_reward += reward

        traj_length = traj_length + 1
        '''Update if its time'''
        if traj_length % args.ppo_update_timestep == 0:
            agent.update()
            traj_length = 0

        if t % args.action_std_decay_freq == 0:
            agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)
        
        if done or episode_timesteps == env.max_episode_step or isinstance(info, ReachGoal):
            if episode_timesteps == env.max_episode_step:
                print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + ', time out: ' + str(episode_timesteps))
            else:
                if isinstance(info, ReachGoal):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + 
                        ', goal reaching at train step: ' + str(episode_timesteps))
                elif isinstance(info, Collision):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + 
                        ', collision occur at train step: ' + str(episode_timesteps))       
                else:
                    raise ValueError('Invalid end signal from environment')
            # Reset environment
            lidar_image, robot_goal_state = env.reset()
            done = False
            episode_num += 1
            writer.add_scalar('train/episode_reward', episode_reward, episode_num)
            episode_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if t % args.eval_freq == 0:
            success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, t,
                                                                     save_directory=file_evaluation_episodes)
            file_name = '/step_' + str(t) + '_success_' + str(int(success_rate * 100))
            print('success_rate, collision_rate, avg_nav_time at step ' + str(t))
            print(success_rate, collision_rate, avg_nav_time)
            writer.add_scalar('eval/success_rate', success_rate, t)
            writer.add_scalar('eval/collision_rate', collision_rate, t)
            evaluations.append(success_rate)
            np.savetxt(file_results + file_name + '.txt', evaluations)
            if success_rate > 0.85 or t % args.save_model_freq == 0:
                agent.save(file_models + file_name + "_ppo.pth")
         
    print('final test')
    success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, t,
                                                             eval_episodes=500, save_directory=file_final_test_episodes,
                                                             final_test=True)
    print('success_rate, collision_rate, avg_nav_time')
    print(success_rate, collision_rate, avg_nav_time)


if __name__ == "__main__":
    main()
