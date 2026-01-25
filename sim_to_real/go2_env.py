import torch
import numpy as np

class Go2Env:
    def __init__(self, device="cuda:0"):
        self.num_envs = 1
        self.num_actions = 12
        self.history_len = 10
        self.num_obs = 45
        self.num_privileged_obs = 286   

        self.device = device

        self.obs = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.history_obs = torch.zeros(self.num_envs, self.history_len * self.num_obs, device=self.device, dtype=torch.float)

    def reset(self, obs):  
        # reset the environment
        self.history_obs = torch.zeros(self.num_envs, self.history_len * self.num_obs, device=self.device, dtype=torch.float)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            obs = obs.astype(np.float32)
        self.obs = torch.from_numpy(obs).to(self.device)
    
    def update_obs(self, obs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            obs = obs.astype(np.float32)
        self.obs = torch.from_numpy(obs).to(self.device)
        history_obs_tmp = self.history_obs[:, self.num_obs:]
        self.history_obs = torch.cat((history_obs_tmp, self.obs), dim=-1)