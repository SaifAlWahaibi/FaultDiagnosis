
import numpy as np


class ReplayBuffer:
    def __init__(self, size, input_shape):

        # attributes: -

        self.size = size
        self.input_shape = input_shape

        self.counter = 0  # counts how many index location are filled in the memory

        # memory: -

        self.state = np.zeros((self.size, self.input_shape), dtype=np.float32)
        self.action = np.zeros(self.size, dtype=np.int64)
        self.reward = np.zeros(self.size, dtype=np.float32)
        self.state_new = np.zeros((self.size, self.input_shape), dtype=np.float32)
        self.terminal = np.zeros(self.size, dtype=np.bool)

    def save(self, state, action, reward, state_new, terminal):
        index = self.counter % self.size  # specifies the next available index location in memory to store data

        # storing data: -

        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.state_new[index] = state_new
        self.terminal[index] = terminal

        self.counter += 1  # specifies how much of the memory is occupied

    def sampler(self, batch):
        memory_index = min(self.counter, self.size)  # identifies the index location of the last stored memory
        batch_index = np.random.choice(memory_index, batch, replace=False)  # randomly chooses data from memory to form
        # a batch of data

        # Extracting the data: -

        state = self.state[batch_index]
        action = self.action[batch_index]
        reward = self.reward[batch_index]
        state_new = self.state_new[batch_index]
        terminal = self.terminal[batch_index]

        return state, action, reward, state_new, terminal
