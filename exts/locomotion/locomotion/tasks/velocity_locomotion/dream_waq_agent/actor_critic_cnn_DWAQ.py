from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class ActorCritic_DWAQ(nn.Module):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, history_len, num_obs, cenet_out_dim, activation="elu", init_noise_std=1.0,):
        super().__init__()

        self.activation = get_activation(activation)
        actor_input_dim = num_actor_obs
        critic_input_dim = num_critic_obs

        self.history_len = history_len
        self.num_obs = num_obs

        self.encoder_out_dim = 64

        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 1)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.Flatten(),
            nn.Linear(64 * 2 * 11, self.encoder_out_dim),
            self.activation,
        )

        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, num_actor_obs - cenet_out_dim)
        )

        print('actor: ', self.actor)
        print('critic: ', self.critic)
        print('encoder: ', self.encoder)
        print('decoder: ', self.decoder)

        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) # this definition may cause NaN value
        self.log_std = nn.Parameter(torch.log(torch.ones(num_actions) * init_noise_std))  # so better with log value
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    
    def cenet_forward(self, obs_history):
        obs_history_reshape = torch.reshape(obs_history.clone(), (-1, 1, self.history_len, self.num_obs))
        distribution = self.encoder(obs_history_reshape)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_mean_vel(distribution)
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # def update_distribution(self, observations):
    #     mean = self.actor(observations)
    #     std = torch.exp(self.log_std)
    #     std = torch.clamp(std, min=1e-3, max=10.0)
    #     self.distribution = Normal(mean, mean * 0.0 + std)

    def update_distribution(self, observations):
        # 1. Protect log_std from NaN infection
        if torch.isnan(self.log_std).any():
            with torch.no_grad():
                self.log_std.nan_to_num_(nan=0.0)

        # 2. Clamped log_std to valid bounds
        clamped_log_std = torch.clamp(self.log_std, min=-8.0, max=2.0)
        std = torch.exp(clamped_log_std)
        std = torch.clamp(std, min=1e-3, max=10.0)
        
        # 3. Predict mean
        observations = torch.nan_to_num(observations, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = self.actor(observations)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)

        # 4. Safe expand
        scale = std.unsqueeze(0).expand_as(mean)
        self.distribution = Normal(mean, scale)

    def act(self, observations, obs_history, **kwargs):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean
    
    def act_encoder_inference(self, observations, obs_history):
        code,_,decode,_,_,_,_ = self.cenet_forward(obs_history)
        observations = torch.cat((code,observations),dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean, code

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
