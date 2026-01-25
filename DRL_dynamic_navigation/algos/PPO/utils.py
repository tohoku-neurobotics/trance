import numpy as np
import torch
import random

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

class RolloutBuffer:
    def __init__(self, device, buffer_size=2000):
        self.buffer_size = buffer_size
        self.device = device
        self.clear()

    def clear(self):
        self.obses, self.robot_goal_states, self.actions = [], [], []
        self.rewards, self.dones = [], []
        self.log_probs, self.values = [], []
        self.returns, self.advantages = [], []

    def add(self, obs, robot_goal_state, action, reward, done, log_prob, value):
        self.obses.append(obs)
        self.robot_goal_states.append(robot_goal_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        values = self.values + [last_value]
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[i])

    def get_batches(self, batch_size=128):
        indices = np.arange(len(self.rewards))
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                'obses': torch.tensor(np.array([self.obses[i] for i in batch_idx]), dtype=torch.float32, device=self.device),
                'robot_goal_states': torch.tensor(np.array([self.robot_goal_states[i] for i in batch_idx]), dtype=torch.float32, device=self.device),
                'actions': torch.tensor(np.array([self.actions[i] for i in batch_idx]), dtype=torch.float32, device=self.device),
                'returns': torch.tensor(np.array([self.returns[i] for i in batch_idx]), dtype=torch.float32, device=self.device),
                'advantages': torch.tensor(np.array([self.advantages[i] for i in batch_idx]), dtype=torch.float32, device=self.device),
                'log_probs': torch.tensor(np.array([self.log_probs[i] for i in batch_idx]), dtype=torch.float32, device=self.device)
            }