
import gym
import numpy as np
from Brain import Agent

if __name__ == "__main__":
    env = gym.make('CartPole-v1')  # initiating the environment

    best_score = -np.inf  # initializing the best score
    load_checkpoint = False  # testing mode

    episodes = 1000  # number of episode or games

    Agent = Agent(gamma=0.99, eps_i=1, eps_d=5e-4, eps_f=0.05, memory_size=100000,
                  input_shape=env.observation_space.shape[0], actions=env.action_space.n, learning_rate=0.001,
                  weight_decay=0.0, batch=64, replace=100, algorithm='Q-Learning', environment='CartPole-v1',
                  directory="C:/Users/SaifA/Documents/Best Model")

    if load_checkpoint:
        Agent.loading_model()

    scores, average_score, eps_history, steps = [], [], [], []

    for i in range(episodes):
        score = 0
        n_steps = 0
        terminal = False
        state = env.reset()[0]  # resetting the environment

        while not terminal:  # until a terminal state is reached
            action = Agent.decision(state)  # choosing an action

            future_state, reward, terminal, truncated, info = env.step(action)  # taking one environmental step in
            # effect of the action

            if truncated:
                terminal = True

            score += reward

            if not load_checkpoint:  # if not testing
                Agent.saving(state=state, action=action, reward=reward, future_state=future_state,
                             terminal=terminal)
                Agent.learn()  # taking a learning step

            state = future_state

            n_steps += 1  # counting steps in an episode

            print('\r', end='')
            print(f'... Learning from Episode {i + 1}, step {n_steps} ...', end='')

        scores.append(score)
        steps.append(n_steps)

        avg_score = np.mean(scores[-100:])  # the average of the last 100 episodes

        average_score.append(avg_score)

        print('\r', end='')
        print('Episode =', i + 1, ': Score = %.3g,' % score, 'Average Score = %.3g' % avg_score)

        if avg_score > best_score:  # saving the best model

            if not load_checkpoint:
                Agent.saving_model()

            best_score = avg_score

        eps_history.append(Agent.eps_i)
