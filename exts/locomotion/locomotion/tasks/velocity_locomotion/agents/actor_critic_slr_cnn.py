# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import torch
import torch.nn as nn
from torch.distributions import Normal

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
def mlp_factory(activation, input_dims, out_dims, hidden_dims, last_act=False):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)

    return layers

class ActorCriticSLR(nn.Module):
    is_recurrent = False

    def __init__(self, num_props, num_hist, num_actions,
                       latent_dims=20,
                       actor_hidden_dims=[256, 256, 256],
                       critic_hidden_dims=[256, 256, 256],
                       mlp_encoder_dims=[256, 128, 64],
                       activation='elu',
                       init_noise_std=1.0):
        super(ActorCriticSLR,self).__init__()

        activation = get_activation(activation)
        self.num_props = num_props
        self.num_hist = num_hist
        self.num_latents = latent_dims

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 2 * 11, latent_dims),
            activation,
        )
        print(self.cnn_encoder)
        
        self.trans = nn.Sequential(nn.Linear(latent_dims + num_actions, 32), nn.ELU(), nn.Linear(32, latent_dims))
        print(self.trans)
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dims + num_props,
                                 out_dims=num_actions,
                                 hidden_dims=actor_hidden_dims))
        print(self.actor)
        
        self.critic = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dims + num_props,
                                 out_dims=1,
                                 hidden_dims=critic_hidden_dims))    
        print(self.critic)

        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) # this definition may cause NaN value
        self.log_std = nn.Parameter(torch.log(torch.ones(num_actions) * init_noise_std))  # so better with log value

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = torch.exp(self.log_std)
        std = torch.clamp(std, min=1e-3, max=10.0)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def extract(self, observations):
        prop = observations[:, :self.num_props]
        hist = observations[:, -self.num_hist * self.num_props:].view(-1, self.num_hist, self.num_props)
        return hist, prop
    
    def encode(self, obs_hist, prop):
        obs = prop
        obs_hist_full = torch.cat([obs_hist[:, 1:, :], obs.unsqueeze(1)], dim=1)
        _, history_len, num_obs = obs_hist_full.size()
        obs_history_reshape = torch.reshape(obs_hist_full.clone(), (-1, 1, history_len, num_obs))
        self.z = self.cnn_encoder(obs_history_reshape)
        return obs, self.z

    def act(self, observations, **kwargs):
        obs_hist, prop = self.extract(observations)
        obs, self.z = self.encode(obs_hist, prop)
        actor_obs = torch.cat([self.z.detach(), obs], dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_hist, prop = self.extract(observations)
        obs, self.z = self.encode(obs_hist, prop)
        actor_obs = torch.cat([self.z.detach(), obs], dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, observations, **kwargs):
        obs_hist, prop = self.extract(observations)
        obs, self.z = self.encode(obs_hist, prop)
        critic_obs = torch.cat([self.z, obs], dim=-1)
        value = self.critic(critic_obs)
        return value
    
    def get_std(self):
        return self.log_std  # initial std has NaN value
    
    def reset(self, dones=None):
        pass

    def save_torch_jit_policy(self):
        encoder_latent_path = 'encoder.jit'
        policy_path = 'policy.jit'
        obs_demo_input = torch.randn(1, self.num_props).to('cpu')
        z_demo_input = torch.randn(1, self.num_latents).to('cpu')
        hist_demo_input = torch.randn(1, self.num_hist * self.num_props).to('cpu')
        
        actor_obs = torch.cat((z_demo_input, obs_demo_input), dim=-1)
        encoder_latent_jit = torch.jit.trace(self.cnn_encoder, hist_demo_input)
        policy_jit = torch.jit.trace(self.actor, actor_obs)
        encoder_latent_jit.save(encoder_latent_path)
        policy_jit.save(policy_path)
        print('-'*50)
        print(f'encoder has been writen in : "{encoder_latent_path}"')
        print(f'policy has been writen in : "{policy_path}"')
        print('-'*50)  
        
