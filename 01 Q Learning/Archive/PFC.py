
import os
import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, name, directory, input_shape, actions, learning_rate, weight_decay):
        super(DeepQNetwork, self).__init__()

        # creating a new path to store critic network: -

        self.directory = directory
        self.directory = os.path.join(self.directory, name)

        # attributes: -

        self.input_shape = input_shape
        self.actions = actions
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # network architecture: -

        self.Dense1 = nn.Linear(self.input_shape, 256)
        self.Act = nn.ReLU()
        self.Dense2 = nn.Linear(256, 256)
        self.Dense3 = nn.Linear(256, self.actions)

        # optimizer: -

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss = nn.MSELoss()  # loss function

        # utilizing GPU if available: -

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):  # the forward pass
        out = self.Act(self.Dense1(state))
        out = self.Act(self.Dense2(out))
        q_values = self.Dense3(out)

        return q_values

    def save_checkpoint(self):
        print('... Saving Checkpoint ...')
        torch.save(self.state_dict(), self.directory)

    def load_checkpoint(self):
        print('... Loading Checkpoint ...')
        self.load_state_dict(torch.load(self.directory))
