import torch
import torch.nn as nn
import torch.nn.functional as F

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
        encoder_feature_dim, action_range
    ):
        super().__init__()

        self.encoder = Encoder(obs_shape, hidden_dim, encoder_feature_dim)

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
        
    def forward(self, feature, terrain, robot_goal_state):
        
        obs = torch.cat((terrain, feature, robot_goal_state), dim=-1)
        
        mu, _ = self.trunk(obs).chunk(2, dim=-1)
    
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return mu
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)


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
        encoder_feature_dim=50,
    ):
        self.device = device
        self.actor = Actor(
            obs_shape, terrain_dim, robot_goal_state_dim, action_dim, hidden_dim,
            encoder_feature_dim, action_range
        ).to(device)

    def get_features(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            feature = self.actor.encoder(obs)
            return feature.cpu().data.numpy().flatten()

    def select_action(self, feature, terrain, robot_goal_state):
        with torch.no_grad():
            feature = torch.FloatTensor(feature).to(self.device)
            feature = feature.unsqueeze(0)
            if terrain is not None:
                terrain = torch.FloatTensor(terrain).to(self.device)
                terrain = terrain.unsqueeze(0)
            robot_goal_state = torch.FloatTensor(robot_goal_state).to(self.device)
            robot_goal_state = robot_goal_state.unsqueeze(0)
            mu = self.actor(feature, terrain, robot_goal_state)
            return mu.cpu().data.numpy().flatten()

    def load(self, filename, device='cuda:0'):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=device))
            
       
       
