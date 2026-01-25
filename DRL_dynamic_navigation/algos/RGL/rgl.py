import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
import numpy as np
import copy
from utils.state import ObservableState, FullState
import torch.optim as optim
from torch.utils.data import DataLoader

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, num_layer, X_dim, wr_dims, wh_dims, final_state_dim,
                 gcn2_w1_dim, planning_dims, similarity_function, layerwise_graph, skip_connection):
        super().__init__()
        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        human_state_dim = input_dim - self_state_dim
        self.self_state_dim = self_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        self.w_r = mlp(self_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)

        if num_layer == 1:
            self.w1 = Parameter(torch.randn(self.X_dim, final_state_dim))
        elif num_layer == 2:
            self.w1 = Parameter(torch.randn(self.X_dim, gcn2_w1_dim))
            self.w2 = Parameter(torch.randn(gcn2_w1_dim, final_state_dim))
        else:
            raise NotImplementedError

        self.value_net = mlp(final_state_dim, planning_dims)

        # for visualization
        self.A = None

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state_input):
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            # lengths = torch.IntTensor([state.size()[1]])

        self_state = state[:, 0, :self.self_state_dim]
        human_states = state[:, :, self.self_state_dim:]

        # compute feature matrix X
        self_state_embedings = self.w_r(self_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([self_state_embedings.unsqueeze(1), human_state_embedings], dim=1)

        # compute matrix A
        normalized_A = self.compute_similarity_matrix(X)
        self.A = normalized_A[0, :, :].data.cpu().numpy()

        # graph convolution
        if self.num_layer == 0:
            feat = X[:, 0, :]
        elif self.num_layer == 1:
            h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            feat = h1[:, 0, :]
        else:
            # compute h1 and h2
            if not self.skip_connection:
                h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1))
            else:
                h1 = relu(torch.matmul(torch.matmul(normalized_A, X), self.w1)) + X
            if self.layerwise_graph:
                normalized_A2 = self.compute_similarity_matrix(h1)
            else:
                normalized_A2 = normalized_A
            if not self.skip_connection:
                h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2))
            else:
                h2 = relu(torch.matmul(torch.matmul(normalized_A2, h1), self.w2)) + h1
            feat = h2[:, 0, :]

        # do planning using only the final layer feature of the agent
        value = self.value_net(feat)
        return value


class RGL:
    def __init__(self, writer, memory, device, action_choices, gamma, time_step, v_pref,
                       optimizer_str='Adam', batch_size=100,
                       num_layer=2, X_dim=32, similarity_function='embedded_gaussian',
                       layerwise_graph=False, skip_connection=True, multiagent_training=True,
                       planning_dims=[150, 100, 100, 1]):
        self.writer = writer
        self.memory = memory
        self.device = device
        self.action_choices = action_choices
        self.gamma = gamma
        self.time_step = time_step
        self.v_pref = v_pref
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.criterion = nn.MSELoss().to(device)
        self.name = 'RGL'
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.epsilon = 0.5
        self.phase = 'train'
        # gcn
        self.multiagent_training = multiagent_training
        wr_dims = [64, X_dim]
        wh_dims = [64, X_dim]
        final_state_dim = X_dim
        gcn2_w1_dim = X_dim
        self.model = ValueNetwork(self.joint_state_dim, self.self_state_dim, num_layer, X_dim, wr_dims, wh_dims,
                                  final_state_dim, gcn2_w1_dim, planning_dims, similarity_function, layerwise_graph,
                                  skip_connection).to(device)
        self.target_model = copy.deepcopy(self.model)
        self.data_loader = None
    
    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_phase(self, phase):
        self.phase = phase

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError

    def propagate(self, state, action, state_type='human'):
        if state_type == 'human':
            # propagate state of humans
            next_px = state.px + action[0] * self.time_step
            next_py = state.py + action[1] * self.time_step
            next_state = ObservableState(next_px, next_py, action[0], action[1], state.radius)
        elif state_type == 'robot':
            # propagate state of current agent
            next_theta = state.theta + action[1] * self.time_step
            if next_theta > np.pi:
                next_theta -= (2.0 * np.pi)
            elif next_theta < -np.pi:
                next_theta += (2.0 * np.pi)
            next_vx = action[0] * np.cos(next_theta)
            next_vy = action[0] * np.sin(next_theta)
            next_px = state.px + next_vx * self.time_step
            next_py = state.py + next_vy * self.time_step
            next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                   state.v_pref, next_theta)
        else:
            raise ValueError('state yype error')

        return next_state

    def predict(self, state):
        if self.reach_destination(state):
            return np.zeros(2)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_choices[np.random.choice(self.action_choices.shape[0])]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            goal_distance_last = np.linalg.norm((state.self_state.px - state.self_state.gx, state.self_state.py - state.self_state.gy))
            for action in self.action_choices:
                next_self_state = self.propagate(state.self_state, action, state_type='robot')
         
                next_human_states = [self.propagate(human_state, np.array([human_state.vx, human_state.vy]))
                                        for human_state in state.human_states]
                reward = self.compute_reward(goal_distance_last, next_self_state, next_human_states)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in next_human_states], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action
    
    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values, _, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss = epoch_loss + loss.data.item()
            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)

        return average_epoch_loss

    def optimize_batch(self, num_batches, total_steps=0):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, _, rewards, next_states = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses = losses + loss.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        self.writer.add_scalar('IL/average_epoch_loss', average_loss, total_steps)

        return average_loss

    def compute_reward(self, goal_distance_last, nav, humans):
        # collision detection
        discomfort_dist = 0.5
        discomfort_penalty_factor = 0.4
        goal_distance_factor = 1.4
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        goal_dist = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy))
        reaching_goal = goal_dist < nav.radius
        dis_goal_reward = goal_distance_factor * (goal_distance_last - goal_dist)

        if (dmin < discomfort_dist):
            dis_obstacle_reward = (dmin - discomfort_dist) * discomfort_penalty_factor
        else:
            dis_obstacle_reward = 0.0

        reward = dis_obstacle_reward + dis_goal_reward 

        if collision:
            reward = -0.5
        elif reaching_goal:
            reward = 0.5

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        rotated_state_tensor = self.rotate(state_tensor)

        return rotated_state_tensor
    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        theta = (state[:, 8] - rot).reshape((batch, -1))
 
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state

    def get_matrix_A(self):
        return self.model.A
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))

def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)
    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    def sort_states(position):
        # sort the sequences in the decreasing order of length
        sequences = sorted([x[position] for x in batch], reverse=True, key=lambda t: t.size()[0])
        packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)

    states = sort_states(0)
    values = torch.cat([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.cat([x[2] for x in batch]).unsqueeze(1)
    next_states = sort_states(3)

    return states, values, rewards, next_states
