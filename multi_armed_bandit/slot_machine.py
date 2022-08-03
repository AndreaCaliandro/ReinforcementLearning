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


class BanditLearningLoop:

    def __init__(self, slot_win_probs, agent, trials=1000):
        self.slot_machines = [SlotMachine(prob) for prob in slot_win_probs]
        self.agent = agent
        self.trials = trials
        self.rewards = np.zeros(trials)
        self.status = {
            'p_estimate': [0] * len(self.slot_machines),
            'num_pulls': [0] * len(self.slot_machines)
        }

    def learning_loop(self):
        for trial in range(self.trials):
            slot_index = self.agent.action(self.status)
            outcome = self.slot_machines[slot_index].pull()
            self.update_status(outcome, slot_index)
            self.update_reward(outcome, trial)

    def update_status(self, outcome, slot_index):
        self.status['num_pulls'][slot_index] += 1
        self.status['p_estimate'][slot_index] += \
            (outcome - self.status['p_estimate'][slot_index]) / self.status['num_pulls'][slot_index]

    def update_reward(self, outcome, trial):
        self.rewards[trial] = outcome
