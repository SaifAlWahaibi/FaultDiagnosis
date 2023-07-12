
import gym
import numpy as np
from Brain import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    top_per = -np.inf

    tst = False

    if tst:
        eps_i = 0.0
        eps_f = 0.0

    else:
        eps_i = 1.0
        eps_f = 1e-4

    esp = 750

    agt = Agent(gamma=0.99, epsilon_initial=eps_i, epsilon_decay=eps_f, epsilon_final=eps_f,
                input_dimensions=env.observation_space.shape[0], action_dimensions=env.action_space.n,
                learning_rate=0.001, weight_decay=0.0, batch_size=64, algorithm='Q-Learning',
                environment=env.spec.id, directory="C:/Users/SaifA/Documents/Best Model")

    if tst:
        agt.loading_model()

    res_arr, avg_res_arr, eps_his, stp = [], [], [], []

    for i in range(esp):
        res = 0
        num_stp = 0
        ter = False
        sta = env.reset()[0]

        while not ter:
            act = agt.decision(sta)

            fut_sta, rwd, ter, tru, inf = env.step(act)

            if tru:
                ter = True

            res += rwd

            if not tst:
                agt.learn(state=sta, action=act, reward=rwd, future_state=fut_sta, terminal=ter)

            sta = fut_sta

            num_stp += 1

            print('\r', end='')
            print(f'... Learning from Episode {i + 1}, step {num_stp} ...', end='')

        res_arr.append(res)
        stp.append(num_stp)

        avg_res = np.mean(res_arr[-100:])
        avg_res_arr.append(avg_res)

        print('\r', end='')
        print('Episode =', i + 1, ': Score = %.3g,' % res, 'Average Score = %.3g' % avg_res)

        if avg_res > top_per:

            if not tst:
                agt.saving_model()

            top_per = avg_res

        eps_his.append(agt.eps_i)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(avg_res_arr)
    plt.title(f'"{env.spec.id}" 100 Runs Average Cumulative Rewards')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    plt.show()
