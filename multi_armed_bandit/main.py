import numpy as np
import matplotlib.pyplot as plt

from multi_armed_bandit.slot_machine import BanditLearningLoop, SlotMachine, GaussianReward
from multi_armed_bandit.agents import (RandomAgent,
                                       GreedyAgent,
                                       OptimisticInitialValuesAgent,
                                       UCB1Agent,
                                       BayesianAgent,
                                       BayesianAgentGaussianReward)


# WINNING_PROBABILITIES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
WINNING_PROBABILITIES = [5.0, 10.0, 20.0]
EPS = 0.1
NUM_TRIALS = 1000


if __name__ == '__main__':
    num_slots = len(WINNING_PROBABILITIES)
    # agent = RandomAgent(num_slots)
    # agent = GreedyAgent(num_slots, epsilon=EPS)
    # agent = OptimisticInitialValuesAgent(num_slots, initial_estimate=10)
    # agent = UCB1Agent(num_slots)
    # agent = BayesianAgent(num_slots)
    agent = BayesianAgentGaussianReward(num_slots)
    print(agent.status)

    slot_machines = [GaussianReward(prob, 1.0) for prob in WINNING_PROBABILITIES]
    # slot_machines = [SlotMachine(prob) for prob in WINNING_PROBABILITIES]
    bandit = BanditLearningLoop(slot_machines, agent, trials=NUM_TRIALS)
    bandit.learning_loop()

    print(agent.status)

    # plot the results
    cumulative_rewards = np.cumsum(agent.rewards)
    print(cumulative_rewards[-1])
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(WINNING_PROBABILITIES))
    plt.xlabel('trial')
    plt.ylabel('win rate')
    plt.show()
