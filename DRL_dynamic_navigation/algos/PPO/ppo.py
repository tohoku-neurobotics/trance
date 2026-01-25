import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.obses = []
        self.robot_goal_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.obses[:]
        del self.robot_goal_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, encoder_feature_dim):
        super(Encoder, self).__init__()
        
        self.obs_shape = obs_shape
        # Convolutional layers for lidar input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 20), stride=(1, 10))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Flatten and dense layers after convolution
        self.flatten_size = self._get_flatten_size(obs_shape)
        self.out_feature = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, encoder_feature_dim)
            )
    
    def _get_flatten_size(self, obs_shape):
        # Function to determine the size after the convolution and pooling
        temp_input = torch.zeros(1, *obs_shape)
        temp_output = self.pool2(self.conv2(self.pool1(self.conv1(temp_input))))
        return temp_output.numel()
    
    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        tie_weights(src=source.conv1, trg=self.conv1)
        tie_weights(src=source.conv2, trg=self.conv2)
    
    def forward(self, lidar_input, detach=False):
        lidar_input = lidar_input.view(lidar_input.size(0), *self.obs_shape)
        
        # Pass through convolutional layers
        x = F.relu(self.conv1(lidar_input))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        if detach:
            x = x.detach()
        
        return self.out_feature(x)

class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, action_dim, hidden_dim, encoder_feature_dim
    ):
        super().__init__()

        self.encoder = Encoder(obs_shape, hidden_dim, encoder_feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim + robot_goal_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )

    def forward(self, obs, robot_goal_state, detach_encoder=True):
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        mu = self.trunk(obs)
        return mu
    
class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, hidden_dim, encoder_feature_dim
    ):
        super().__init__()

        self.encoder = Encoder(obs_shape, hidden_dim, encoder_feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(robot_goal_state_dim + encoder_feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, robot_goal_state, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        v = self.trunk(obs)

        return v


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, robot_goal_state_dim, action_range, device,
                 action_dim, hidden_dim, encoder_feature_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_scale = torch.FloatTensor((action_range[1] - action_range[0]) / 2.0).to(device)
        self.action_bias = torch.FloatTensor((action_range[1] + action_range[0]) / 2.0).to(device)
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        self.actor = Actor(obs_shape, robot_goal_state_dim, action_dim, hidden_dim, encoder_feature_dim)
      
        # critic
        self.critic = Critic(obs_shape, robot_goal_state_dim, hidden_dim, encoder_feature_dim)
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, obs, robot_goal_state):
        action_mean = self.actor(obs, robot_goal_state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        raw_action = dist.sample()
        scaled_action = self.action_scale * raw_action + self.action_bias

        action_logprob = dist.log_prob(action)
        state_val = self.critic(obs, robot_goal_state)

        return raw_action.detach(), scaled_action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, obses, robot_goal_states, action):
        raw_action = (action - self.action_bias) / self.action_scale
        action_mean = self.actor(obses, robot_goal_states)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            raw_action = raw_action.reshape(-1, self.action_dim)
  
        action_logprobs = dist.log_prob(raw_action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obses, robot_goal_states)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, obs_shape, robot_goal_state_dim, action_dim, action_range, device,
                 encoder_feature_dim=50,
                 hidden_dim=256,
                 lr=0.0003,
                 gamma=0.99, 
                 K_epochs=80, 
                 eps_clip=0.2, 
                 action_std_init=0.6):
        self.action_std = action_std_init
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(obs_shape, robot_goal_state_dim, action_range, device, action_dim, hidden_dim, 
                                  encoder_feature_dim, action_std_init).to(device)
        # tie encoders between actor and critic
        self.policy.actor.encoder.copy_conv_weights_from(self.policy.critic.encoder)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(obs_shape, robot_goal_state_dim, action_range, device, action_dim, hidden_dim, 
                                      encoder_feature_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
  
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, obs, robot_goal_state):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            robot_goal_state_tensor = torch.FloatTensor(robot_goal_state).to(self.device)
            obs_tensor = obs_tensor.unsqueeze(0)
            robot_goal_state_tensor = robot_goal_state_tensor.unsqueeze(0)
            raw_action, scaled_action, action_logprob, state_val = self.policy_old.act(obs_tensor, robot_goal_state_tensor)

        self.buffer.obses.append(obs_tensor)
        self.buffer.robot_goal_states.append(robot_goal_state_tensor)
        self.buffer.actions.append(raw_action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return scaled_action.detach().cpu().numpy().flatten()
    
    def evaluation_action(self, obs, robot_goal_state):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            robot_goal_state_tensor = torch.FloatTensor(robot_goal_state).to(self.device)
            obs_tensor = obs_tensor.unsqueeze(0)
            robot_goal_state_tensor = robot_goal_state_tensor.unsqueeze(0)
            action_mean = self.policy.actor(obs_tensor, robot_goal_state_tensor)
            scaled_action = self.policy.action_scale * action_mean + self.policy.action_bias

        return scaled_action.detach().cpu().numpy().flatten()
     
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_obses = torch.squeeze(torch.stack(self.buffer.obses, dim=0)).detach().to(self.device)
        old_robot_goal_states = torch.squeeze(torch.stack(self.buffer.robot_goal_states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_obses, old_robot_goal_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

