
import os
import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, network_name, directory, input_dimensions, action_dimensions, learning_rate, weight_decay):
        super(DeepQNetwork, self).__init__()

        self.pth = os.path.join(directory, network_name)

        self.inp_dim = input_dimensions
        self.act_dim = action_dimensions
        self.lr = learning_rate
        self.l2_reg = weight_decay

        self.q_val = nn.Sequential(nn.Linear(self.inp_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, self.act_dim)
                                   )

        self.opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        self.lss = nn.MSELoss()

        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.dev)

    def forward(self, state):
        q_values = self.q_val(state)

        return q_values

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.pth)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.pth))
