
from Hippocampus import ReplayBuffer
from PFC import DeepQNetwork
import torch as torch
import numpy as np


class Agent:
    def __init__(self, gamma, eps_i, eps_d, eps_f, memory_size, input_shape, actions, learning_rate, weight_decay,
                 batch, replace, algorithm, environment, directory):

        # attributes: -

        self.gamma = gamma
        self.eps_i = eps_i
        self.eps_d = eps_d
        self.eps_f = eps_f
        self.memory_size = memory_size
        self.input_shape = input_shape
        self.actions = actions
        self.action_space = [i for i in range(actions)]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch = batch
        self.replace = replace
        self.algorithm = algorithm
        self.environment = environment
        self.directory = directory
        self.learning_step = 0  # counter to keep track of how many learning steps are taken

        self.memory = ReplayBuffer(self.memory_size, self.input_shape)  # initializing the memory

        # initializing the Q-networks: -

        self.Q_eval = DeepQNetwork(name=self.environment+'_'+self.algorithm+'Double_Q_eval.pth',
                                   directory=self.directory, input_shape=self.input_shape, actions=self.actions,
                                   learning_rate=self.learning_rate, weight_decay=self.weight_decay)

    def decision(self, observation):
        if np.random.random() > self.eps_i:  # deciding to explore or exploit
            self.Q_eval.eval()

            with torch.no_grad():
                state = torch.tensor(observation, dtype=torch.float).to(self.Q_eval.device)  # saving state values in
                # the same memory as the model
                q_values = self.Q_eval.forward(state)  # calculating q-values
                action = torch.argmax(q_values).item()  # choosing the action associated with the highest q_values

            self.Q_eval.train()
        else:
            action = np.random.choice(self.action_space)  # randomly choosing actions

        return action

    def saving(self, state, action, reward, future_state, terminal):
        self.memory.save(state, action, reward, future_state, terminal)  # storing data in memory

    def sampling(self):
        state, action, reward, future_state, terminal = self.memory.sampler(self.batch)  # sampling memory

        # saving values in the same memory as the model: -

        state = torch.tensor(state).to(self.Q_eval.device)
        action = torch.tensor(action).to(self.Q_eval.device)
        reward = torch.tensor(reward).to(self.Q_eval.device)
        future_state = torch.tensor(future_state).to(self.Q_eval.device)
        terminal = torch.tensor(terminal).to(self.Q_eval.device)

        return state, action, reward, future_state, terminal

    def epsilon(self):

        if self.eps_i > self.eps_f:  # decrementing epsilon
            self.eps_i = self.eps_i - self.eps_d

        else:
            self.eps_i = self.eps_f

    def saving_model(self):  # saving Q-networks
        self.Q_eval.save_checkpoint()

    def loading_model(self):  # loading saved Q-networks
        self.Q_eval.load_checkpoint()

    def learn(self):

        if self.memory.counter < self.batch:  # skipping learning if the memory is not sufficiently filled
            return

        self.Q_eval.optimizer.zero_grad()  # zeroing the gradient

        states, actions, rewards, future_states, terminals = self.sampling()  # sampling memory to collect a batch
        # worth of data

        batch_index = np.arange(self.batch, dtype=np.int32)
        q_prediction = self.Q_eval(states)[batch_index, actions]  # predicting the q_values associated with the actions
        # specifically taken as per the sampled memory batch

        self.Q_eval.eval()

        with torch.no_grad():
            q_future = self.Q_eval(future_states)  # estimating target q_value

        self.Q_eval.train()

        q_future[terminals] = 0.0   # setting the q_values for terminal states to zero

        future_action = torch.argmax(self.Q_eval(future_states), dim=1)  # estimating future action

        td_target = rewards + self.gamma * q_future[batch_index, future_action]  # calculating TD target

        loss = self.Q_eval.loss(td_target, q_prediction).to(self.Q_eval.device)  # evaluating the loss function
        loss.backward()  # back-prop
        self.Q_eval.optimizer.step()  # updating weights
        self.learning_step += 1

        self.epsilon()  # decrementing
