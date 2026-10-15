import argparse
import numpy as np
import random
import os
import torch

from algos.RGL.memory import ReplayMemory
from algos.RGL.rgl import RGL
from info import *
from env_rgl import CrowdSim
from torch.utils.tensorboard import SummaryWriter

import time

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_or_test_env, current_steps, 
                eval_episodes=100, save_directory=None, 
                final_test=False):
    policy.set_phase('eval')
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
    seed_all = [23, 24, 37, 38, 42, 47, 49, 59, 61]
    seed_specific = seed_all[8]

    for i in range(eval_episodes):
        if_save_data = (i < 10 or final_test)
        if final_test:
            if eval_episodes == 1:
                # please set your own seed to randomize a specific
                state = eval_or_test_env.reset(seed=seed_specific, save_data=if_save_data)
            else:
                state = eval_or_test_env.reset(seed=i, save_data=if_save_data)
        else:
            state  = eval_or_test_env.reset(save_data=if_save_data)
        # eval_or_test_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_or_test_env.max_episode_step:
            # t1 = time.time()
            action = policy.predict(state)
            # action = eval_or_test_env.dwa_compute_action()
            state, reward, done, info = eval_or_test_env.step(action, eval=True, save_data=if_save_data)
            # print('time: ', time.time() - t1) # 7.5ms
            # eval_or_test_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            ep_step = ep_step + 1
            if done or ep_step == eval_or_test_env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == eval_or_test_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_or_test_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))         
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        if save_directory is not None:
            if not isinstance(info, ReachGoal):
                if eval_episodes == 1:
                    file_name = save_directory + '/eval_' + str(seed_specific) + '_' + str(i) + '_fail' + '.npz'
                else:
                    file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
            else:
                if eval_episodes == 1:
                    file_name = save_directory + '/eval_' + str(seed_specific) + '_' + str(i) + '.npz'
                else:
                    file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '.npz'
            if if_save_data:
                np.savez_compressed(file_name, **eval_or_test_env.log_env)

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_or_test_env.time_limit

    return success_rate, collision_rate, avg_nav_time

def collect_samples(k, memory, policy, env, gamma, device, imitation_learning=False, episode=0):
    policy.set_phase('train')
    success = 0
    collision = 0
    timeout = 0
    success_times = []
    num_samples = 0
    for i in range(k):
        state = env.reset()
        states = []
        actions = []
        rewards = []
        done = False
        ep_step = 0
        while ep_step < env.max_episode_step:
            if imitation_learning:
                action = env.cal_orca_action()
            else:
                action = policy.predict(state)
            state, reward, done, info = env.step(action)
            if imitation_learning:
                states.append(env.robot_policy.last_state)
            else:
                states.append(policy.last_state)
            actions.append(action)
            rewards.append(reward)

            ep_step = ep_step + 1
            if done or ep_step == env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == env.max_episode_step:
                    timeout += 1
                    if not imitation_learning:
                        print('sampling episode ' + str(i + episode) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(env.global_time)
                        if not imitation_learning:
                            print('sampling episode ' + str(i + episode) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(info, Collision):
                        collision += 1
                        if not imitation_learning:
                            print('sampling episode ' + str(i + episode) + ', collision occur at evaluation step: ' + str(ep_step))         
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        num_samples = num_samples + ep_step
        if isinstance(info, ReachGoal) or isinstance(info, Collision):
            # only add positive(success) or negative(collision) experience in experience set
            for i, state in enumerate(states[:-1]):
                reward = rewards[i]

                # VALUE UPDATE
                if imitation_learning:
                    # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                    state = policy.transform(state)
                    next_state = policy.transform(states[i + 1])
                    value = sum([pow(gamma, (t - i) * env.time_step * env.robot.v_pref) * reward *
                                (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                else:
                    next_state = states[i + 1]
                    if i == len(states) - 1:
                        # terminal state
                        value = reward
                    else:
                        value = 0
                value = torch.Tensor([value]).to(device)
                reward = torch.Tensor([rewards[i]]).to(device)
               
                memory.push((state, value, reward, next_state))

    if imitation_learning:
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit
        print('success_rate, collision_rate, avg_nav_time at episode ' + str(episode))
        print(success_rate, collision_rate, avg_nav_time)
    return num_samples

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--policy', default='rgl', type=str)
    # differential
    parser.add_argument('--robot_model', default='differential', type=str)
    parser.add_argument('--use_angular', action='store_true', default=False)
    parser.add_argument('--random_radius', action='store_true', default=True)
    
    # device
    parser.add_argument('--device', type=str, default='cuda')
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--seed', default=4, type=int)

    # How often (episodes) we evaluate
    parser.add_argument('--eval_freq', default=200, type=int)
    # How often (episodes) we save the trained model
    parser.add_argument('--save_model_freq', default=1000, type=int)
    # Max episodes to run environment
    parser.add_argument('--train_episodes', default=20000, type=int)
    parser.add_argument('--sample_episodes', default=1, type=int)
    parser.add_argument('--target_update_interval', default=1000, type=int)
    
    # number of batches to train at the end of training episode il_episodes
    parser.add_argument('--train_batches', default=100, type=int)
    parser.add_argument('--epsilon_decay', default=16000, type=int)
    parser.add_argument("--epsilon_start", type=float, default=0.8)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    # replay buffer
    parser.add_argument("--capacity", default=100000, type=int)
    # RL gamman
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--il_episodes", default=2000, type=int)
    parser.add_argument("--il_epochs", default=50, type=int)
    parser.add_argument("--il_learning_rate", type=float, default=0.001)
    parser.add_argument("--rl_learning_rate", type=float, default=0.001)

    # Model load file name, "" doesn't load, "default" uses file_name
    # args.load_model, format, step_NO_success_NO
    # parser.add_argument("--load_model", type=str, default="step_2740000_success_90")
    parser.add_argument("--load_model", type=str, default="")

    # parser.add_argument("--load_test_model", type=str, default="")
    parser.add_argument("--load_test_model", type=str, default="step_1478485_success_87")
    parser.add_argument('--test_single', action='store_true', default=False)
    # environment settings
    parser.add_argument("--human_num_max", type=int, default=4)
    parser.add_argument("--static_obstacle_num_max", type=int, default=3)
    parser.add_argument("--square_width", type=float, default=10.0)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, Seed: {args.seed}")
    print("---------------------------------------")
    
    file_prefix = './logs/' + args.policy + '_' + args.robot_model
    if args.use_angular:
        file_prefix = file_prefix + '_angular'
    if args.random_radius:
        file_prefix = file_prefix + '_random_radius'

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
    
    action_resolution = [0.1, 0.2]
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

    env = CrowdSim(args, action_range, action_choices=action_choices)    
    eval_env = CrowdSim(args, action_range, action_choices=action_choices)
    set_seed_everywhere(args.seed)

    memory = ReplayMemory(args.capacity)

    device = torch.device(args.device)
    agent = RGL(writer, memory, device, action_choices, args.gamma, env.time_step, env.robot.v_pref)

    total_steps = 0
    if args.load_model != "":
        agent.load(file_models + '/' + args.load_model)
        # args.load_model, format, step_NO_success_NO
        # extract the first number
        total_steps = int(args.load_model.split('_')[1])

    if args.load_test_model != "":
        print('start to test')
        agent.load(file_models + '/' + args.load_test_model)
        if args.test_single:
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
    if args.load_model == "":
        # start imitation learning
        agent.set_learning_rate(args.il_learning_rate)
        collect_samples(args.il_episodes, memory, agent,
                        env, args.gamma, device, imitation_learning=True)
        agent.optimize_epoch(args.il_epochs)
        il_weight_file = file_models + '/il_weight'
        agent.save(il_weight_file)
        agent.update_target_model()
        print('Evaluate the model instantly after imitation learning on the validation cases')
        success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, total_steps, 
                                                                 save_directory=file_evaluation_episodes)
        file_name = '/step_' + str(total_steps) + '_success_' + str(int(success_rate * 100))
        print('success_rate, collision_rate, avg_nav_time at step ' + str(total_steps))
        print(success_rate, collision_rate, avg_nav_time)
        writer.add_scalar('eval/success_rate', success_rate, total_steps)
        writer.add_scalar('eval/collision_rate', collision_rate, total_steps)
        total_steps_success_rate = np.array([total_steps, success_rate], dtype=np.float32)
        evaluations.append(total_steps_success_rate)
        np.savetxt(file_results + file_name + '.txt', np.array(evaluations))

    agent.set_learning_rate(args.rl_learning_rate)
    current_episode = 0
    while current_episode < args.train_episodes:
        if current_episode < args.epsilon_decay:
            epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) / args.epsilon_decay * current_episode
        else:
            epsilon = args.epsilon_end
        agent.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        num_samples = collect_samples(args.sample_episodes, memory, agent, env, args.gamma, device, episode=current_episode)
        total_steps = total_steps + num_samples

        agent.optimize_batch(args.train_batches, total_steps)
        current_episode = current_episode + 1

        if current_episode % args.target_update_interval == 0:
            print('update target model')
            agent.update_target_model()
        # evaluate the model
        if current_episode % args.eval_freq == 0:
            print('evaluate model')
            success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, total_steps, 
                                                                     save_directory=file_evaluation_episodes)
            file_name = '/step_' + str(total_steps) + '_success_' + str(int(success_rate * 100))
            print('success_rate, collision_rate, avg_nav_time at step ' + str(total_steps))
            print(success_rate, collision_rate, avg_nav_time)
            writer.add_scalar('eval/success_rate', success_rate, total_steps)
            writer.add_scalar('eval/collision_rate', collision_rate, total_steps)
            total_steps_success_rate = np.array([total_steps, success_rate], dtype=np.float32)
            evaluations.append(total_steps_success_rate)
            np.savetxt(file_results + file_name + '.txt', np.array(evaluations))
            if success_rate > 0.85 or current_episode % args.save_model_freq == 0:
                agent.save(file_models + file_name)
         
    print('final test')
    success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, total_steps, 
                                                             eval_episodes=500, save_directory=file_final_test_episodes,
                                                             final_test=True)
    print('success_rate, collision_rate, avg_nav_time')
    print(success_rate, collision_rate, avg_nav_time)


if __name__ == "__main__":
    main()