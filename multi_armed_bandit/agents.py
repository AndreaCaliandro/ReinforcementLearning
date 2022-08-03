import numpy as np


class BaseAgent:

    def __init__(self, num_slots):
        self.num_slots = num_slots

    def action(self, status):
        pass


class RandomAgent(BaseAgent):

    def action(self, status):
        return np.random.randint(self.num_slots)


class GreedyAgent(BaseAgent):

    def __init__(self, num_slots, epsilon):
        super().__init__(num_slots)
        self.epsilon = epsilon

    def action(self, status):
        if (np.random.uniform() < self.epsilon) or (np.array(status['num_pulls']).sum() == 0):
            return np.random.randint(self.num_slots)
        return np.argmax(status['p_estimate'])
