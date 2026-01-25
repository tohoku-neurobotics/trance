import torch
import numpy as np
import random


class eval_policy_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, terrain_dim, robot_goal_state_dim, action_dim, 
                 capacity=200000, batch_size=128, device='cuda:0'):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.terrain_dim = terrain_dim
        obs_dtype = np.float32

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        if self.terrain_dim > 0:
            self.terrains = np.empty((capacity, terrain_dim), dtype=obs_dtype)
        self.robot_goal_states = np.empty((capacity, robot_goal_state_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        if self.terrain_dim > 0:
            self.next_terrains = np.empty((capacity, terrain_dim), dtype=obs_dtype)
        self.next_robot_goal_states = np.empty((capacity, robot_goal_state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, terrain, robot_goal_state, action, reward, 
                  next_obs, next_terrain, next_robot_goal_state, done):
        np.copyto(self.obses[self.idx], obs)
        if self.terrain_dim > 0:
            np.copyto(self.terrains[self.idx], terrain)
        np.copyto(self.robot_goal_states[self.idx], robot_goal_state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        if self.terrain_dim > 0:
            np.copyto(self.next_terrains[self.idx], next_terrain)
        np.copyto(self.next_robot_goal_states[self.idx], next_robot_goal_state)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        if self.terrain_dim > 0:
            terrains = torch.as_tensor(self.terrains[idxs], device=self.device)
        else:
            terrains = None
        robot_goal_states = torch.as_tensor(self.robot_goal_states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        if self.terrain_dim > 0:
            next_terrains = torch.as_tensor(self.next_terrains[idxs], device=self.device)
        else:
            next_terrains = None
        next_robot_goal_states = torch.as_tensor(self.next_robot_goal_states[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, terrains, robot_goal_states, actions, rewards, next_obses, next_terrains, next_robot_goal_states, not_dones




