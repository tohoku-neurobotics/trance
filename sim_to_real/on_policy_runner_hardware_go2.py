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

from actor_critic_cnn_DWAQ import ActorCritic_DWAQ


class OnPolicyRunner:

    def __init__(self,
                 num_privileged_obs, num_obs, num_actions, history_len, 
                 device='cuda:0'):

        self.device = device
        num_critic_obs = num_privileged_obs 
    
        cenet_out_dim = 19
        self.actor_critic = ActorCritic_DWAQ(num_obs + cenet_out_dim,
                                             num_critic_obs,
                                             num_actions,
                                             history_len, num_obs,
                                             cenet_out_dim).to(self.device)

    def load(self, path):
        loaded_dict = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
    
    def get_inference_policy_encoder(self, device=None):
        self.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_encoder_inference
