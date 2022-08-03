import numpy as np
import matplotlib.pyplot as plt

from multi_armed_bandit.slot_machine import BanditLearningLoop
from multi_armed_bandit.agents import RandomAgent, GreedyAgent


WINNING_PROBABILITIES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
EPS = 0.1
NUM_TRIALS = 1000


if __name__ == '__main__':
    agent = RandomAgent(len(WINNING_PROBABILITIES))
    # agent = GreedyAgent(len(WINNING_PROBABILITIES), epsilon=EPS)

    bandit = BanditLearningLoop(WINNING_PROBABILITIES, agent, trials=NUM_TRIALS)
    bandit.learning_loop()

    print(bandit.status)

    # plot the results
    cumulative_rewards = np.cumsum(bandit.rewards)
    print(cumulative_rewards[-1])
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(WINNING_PROBABILITIES))
    plt.xlabel('trial')
    plt.ylabel('win rate')
    plt.show()
