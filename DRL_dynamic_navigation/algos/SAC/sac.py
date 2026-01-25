import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from algos.SAC.utils import soft_update_params

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

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
        self, obs_shape, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim,
        encoder_feature_dim, log_std_min, log_std_max, action_range
    ):
        super().__init__()

        self.encoder = Encoder(obs_shape, hidden_dim, encoder_feature_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(terrain_dim + encoder_feature_dim + robot_goal_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )
        
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (action_range[1] - action_range[0]) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_range[1] + action_range[0]) / 2.)
        
    def forward(
        self, obs, terrain, robot_goal_state, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)
        
        if terrain is not None:
            obs = torch.cat((terrain, obs, robot_goal_state), dim=-1)
        else:
            obs = torch.cat((obs, robot_goal_state), dim=-1)
        
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        std = log_std.exp()

        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        pi = y_t * self.action_scale + self.action_bias
        log_pi = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_pi -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_pi = log_pi.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return mu, pi, log_pi, log_std
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(terrain_dim + obs_dim + robot_goal_state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim, encoder_feature_dim
    ):
        super().__init__()

        self.encoder = Encoder(obs_shape, hidden_dim, encoder_feature_dim)

        self.Q1 = QFunction(
            encoder_feature_dim, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim
        )
        self.Q2 = QFunction(
            encoder_feature_dim, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim
        )

    def forward(self, obs, terrain, robot_goal_state, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        if terrain is not None:
            obs = torch.cat((terrain, obs, robot_goal_state), dim=-1)
        else:
            obs = torch.cat((obs, robot_goal_state), dim=-1)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2

class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        terrain_dim,
        robot_goal_state_dim, 
        action_dim,
        action_range, 
        device,
        hidden_dim=1024,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_beta=0.5,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.01,
        critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_tau=0.05,
    ):
        self.terrain_dim = terrain_dim
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        self.actor = Actor(
            obs_shape, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max, action_range
        ).to(device)

        self.critic = Critic(
            obs_shape, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim, encoder_feature_dim
        ).to(device)

        self.critic_target = Critic(
            obs_shape, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim, encoder_feature_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
       

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, terrain, robot_goal_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            if terrain is not None:
                terrain = torch.FloatTensor(terrain).to(self.device)
                terrain = terrain.unsqueeze(0)
            robot_goal_state = torch.FloatTensor(robot_goal_state).to(self.device)
            robot_goal_state = robot_goal_state.unsqueeze(0)
            mu, _1, _2, _3 = self.actor(obs, terrain, robot_goal_state)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, terrain, robot_goal_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            if terrain is not None:
                terrain = torch.FloatTensor(terrain).to(self.device)
                terrain = terrain.unsqueeze(0)
            robot_goal_state = torch.FloatTensor(robot_goal_state).to(self.device)
            robot_goal_state = robot_goal_state.unsqueeze(0)
            _1, pi, _2, _3 = self.actor(obs, terrain, robot_goal_state)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, terrain, robot_goal_state, action, reward, 
                            next_obs, next_terrain, next_robot_goal_state, not_done, writer, step):
        with torch.no_grad():
            _1, policy_action, log_pi, _2 = self.actor(next_obs, next_terrain, next_robot_goal_state)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_terrain, next_robot_goal_state, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, terrain, robot_goal_state, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        writer.add_scalar('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, terrain, robot_goal_state, writer, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, terrain, robot_goal_state, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, terrain, robot_goal_state, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        writer.add_scalar('train_actor/loss', actor_loss, step)
        writer.add_scalar('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        writer.add_scalar('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        writer.add_scalar('train_alpha/loss', alpha_loss, step)
        writer.add_scalar('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, writer, step):
        obs, terrain, robot_goal_state, action, reward, \
            next_obs, next_terrain, next_robot_goal_state, not_done = replay_buffer.sample()

        writer.add_scalar('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, terrain, robot_goal_state, action, reward, 
                           next_obs, next_terrain, next_robot_goal_state, not_done, writer, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, terrain, robot_goal_state, writer, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_target.state_dict(), filename + '_critic_target')
        torch.save(self.actor_optimizer.state_dict(),
                   filename + '_actor_optimizer')
        torch.save(self.critic_optimizer.state_dict(),
                   filename + '_critic_optimizer')
    
    def load_partial_model(self, old_agent):
        for param in self.actor.parameters():
            param.data.zero_()
        for param in self.critic.parameters():
            param.data.zero_()
        for param in self.critic_target.parameters():
            param.data.zero_()
        soft_update_params(old_agent.actor.encoder, self.actor.encoder, 1.0)
        old_model_trunk = [old_agent.actor.trunk.state_dict()]
        new_model_trunk = [self.actor.trunk.state_dict()]
    
        soft_update_params(old_agent.critic.encoder, self.critic.encoder, 1.0)
        old_model_trunk.append(old_agent.critic.Q1.trunk.state_dict())
        old_model_trunk.append(old_agent.critic.Q2.trunk.state_dict())
        new_model_trunk.append(self.critic.Q1.trunk.state_dict())
        new_model_trunk.append(self.critic.Q2.trunk.state_dict())
    
        soft_update_params(old_agent.critic_target.encoder, self.critic_target.encoder, 1.0)
        old_model_trunk.append(old_agent.critic_target.Q1.trunk.state_dict())
        old_model_trunk.append(old_agent.critic_target.Q2.trunk.state_dict())
        new_model_trunk.append(self.critic_target.Q1.trunk.state_dict())
        new_model_trunk.append(self.critic_target.Q2.trunk.state_dict())
   
        # Handle each layer in trunk
        trunk_num = len(old_model_trunk)
        for i in range(trunk_num):
            old_state_dict = old_model_trunk[i]
            new_state_dict = new_model_trunk[i]
            for name, param in old_state_dict.items():
                if name not in new_state_dict:
                    continue
                old_param = param
                new_param = new_state_dict[name]
                
                if old_param.shape == new_param.shape:
                    new_param.copy_(old_param)
                else:
                    # Partial copy only if shapes are compatible (e.g., Linear weights or bias)
                    if '0.weight' in name:  # first Linear layer (input layer)
                        new_param[:, self.terrain_dim:].copy_(old_param)
                        print(f"Partially copied {name} — {old_param.shape} → {new_param.shape}")
                    elif '0.bias' in name:
                        new_param.copy_(old_param)
                    else:
                        print(f"Skipping {name} due to shape mismatch: {old_param.shape} vs {new_param.shape}")

    def load(self, filename, old_agent=None, device='cuda:0'):
        if old_agent is not None:
            self.load_partial_model(old_agent)
        else:
            self.actor.load_state_dict(torch.load(filename + '_actor', map_location=device))
            self.critic.load_state_dict(torch.load(filename + '_critic', map_location=device))
            self.critic_target.load_state_dict(torch.load(filename + '_critic_target', map_location=device))
            self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer', map_location=device))
            self.critic_optimizer.load_state_dict(torch.load(filename + '_critic_optimizer', map_location=device))
       
       
