
from PFC import DeepQNetwork
import torch as torch
import numpy as np


class Agent:
    def __init__(self, gamma, epsilon_initial, epsilon_decay, epsilon_final, input_dimensions, action_dimensions,
                 learning_rate, weight_decay, batch_size, algorithm, environment, directory):
        self.gam = gamma
        self.eps_i = epsilon_initial
        self.eps_d = epsilon_decay
        self.eps_f = epsilon_final
        self.inp_dim = input_dimensions
        self.act = action_dimensions
        self.lr = learning_rate
        self.l2_reg = weight_decay
        self.bat_dim = batch_size
        self.alg = algorithm
        self.env = environment
        self.pth = directory

        self.act_int = [i for i in range(action_dimensions)]

        self.q_net = DeepQNetwork(network_name=self.env + '_' + self.alg + '_' + 'Q-Network', directory=self.pth,
                                  input_dimensions=self.inp_dim, action_dimensions=self.act, learning_rate=self.lr,
                                  weight_decay=self.l2_reg)

    def decision(self, state):
        if np.random.random() > self.eps_i:
            self.q_net.eval()

            with torch.no_grad():
                sta = torch.tensor(state, dtype=torch.float).to(self.q_net.dev)
                q_val = self.q_net.forward(sta)
                action = torch.argmax(q_val).item()

            self.q_net.train()
        else:
            action = np.random.choice(self.act_int)

        return action

    def epsilon_update(self):

        if self.eps_i > self.eps_f:
            self.eps_i = self.eps_i - self.eps_d

        else:
            self.eps_i = self.eps_f

    def saving_model(self):
        print('... Saving Model ...')
        self.q_net.save_checkpoint()

    def loading_model(self):
        print('... Loading Model ...')
        self.q_net.load_checkpoint()

    def learn(self, state, action, reward, future_state, terminal):
        sta = torch.tensor(state).to(self.q_net.dev)
        rwd = torch.tensor(reward).to(self.q_net.dev)
        fut_sta = torch.tensor(future_state).to(self.q_net.dev)

        self.q_net.opt.zero_grad()

        q_prd = self.q_net(sta)[action]

        self.q_net.eval()

        with torch.no_grad():
            q_fut = torch.max(self.q_net(fut_sta), dim=-1)[0]

        self.q_net.train()

        td_tar = rwd + (1 - int(terminal)) * self.gam * q_fut

        lss = self.q_net.lss(td_tar, q_prd).to(self.q_net.dev)
        lss.backward()
        self.q_net.opt.step()

        self.epsilon_update()
