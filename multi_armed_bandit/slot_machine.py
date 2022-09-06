import numpy as np


class SlotMachine:
    """
    Simulate a slot machine with static winning probability
    """

    def __init__(self, winning_prob):
        self.winning_prob = winning_prob

    def pull(self):
        """
        Return the outcome of your pull. True for win, false for loose
        """
        chance = np.random.uniform()
        return chance < self.winning_prob


class GaussianReward:

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def pull(self):
        return np.random.normal(self.mean, self.sigma)


class BanditLearningLoop:

    def __init__(self, slot_machine_list, agent, trials=1000):
        self.slot_machines = slot_machine_list
            # [SlotMachine(prob) for prob in slot_win_probs]
        self.agent = agent
        self.trials = trials

    def learning_loop(self):
        for trial in range(self.trials):
            slot_index = self.agent.action()
            outcome = self.slot_machines[slot_index].pull()
            self.agent.update_status(outcome, slot_index)
            self.agent.update_reward(outcome)
