import numpy as np
from scipy.stats import beta


class BaseAgent:

    def __init__(self, num_slots):
        self.num_slots = num_slots
        self.p_estimate = [0] * self.num_slots
        self.rewards = []
        self.status = {
            'p_estimate': [0] * self.num_slots,
            'num_pulls': [0] * self.num_slots
        }

    def action(self):
        return np.argmax(self.status['p_estimate'])

    def update_status(self, outcome, slot_index):
        self.status['num_pulls'][slot_index] += 1
        self.status['p_estimate'][slot_index] += \
            (outcome - self.status['p_estimate'][slot_index]) / self.status['num_pulls'][slot_index]

    def update_reward(self, outcome):
        self.rewards.append(outcome)


class RandomAgent(BaseAgent):

    def action(self):
        return np.random.randint(self.num_slots)


class GreedyAgent(BaseAgent):

    def __init__(self, num_slots, epsilon):
        super().__init__(num_slots)
        self.epsilon = epsilon

    def action(self):
        if (np.random.uniform() < self.epsilon) or (np.array(self.status['num_pulls']).sum() == 0):
            return np.random.randint(self.num_slots)
        return np.argmax(self.status['p_estimate'])


class OptimisticInitialValuesAgent(BaseAgent):
    """
    We set the initial value of the probability estimate to an high value.
    The effect is that in the first ohase of the experiment every time a Slot is selected,
    its p_estimate will decrease regardless of the outcome.
    Such Slot will not be selected untill the estimate of the others will go below it.
    In this way, exploration is guarantee without the need to define the epsilon.
    """
    def __init__(self, num_slots, initial_estimate):
        super().__init__(num_slots)
        self.status = {
            'p_estimate': [initial_estimate] * self.num_slots,
            'num_pulls': [1] * self.num_slots
        }


class UCB1Agent(BaseAgent):
    """
    Implement the algorithm Upper Confident Bound 1
    """
    def __init__(self, num_slots):
        super().__init__(num_slots)
        self.status = {
            'mean': np.array([0.0] * self.num_slots),
            'upper_bound': np.array([np.sqrt(2 * np.log(self.num_slots))] * self.num_slots),
            'num_pulls': np.array([1] * self.num_slots)
        }
        self.status.update({'p_estimate': self.status['mean'] + self.status['upper_bound']})

    def update_status(self, outcome, slot_index):
        self.status['num_pulls'][slot_index] += 1
        self.status['mean'][slot_index] += \
            (outcome - self.status['mean'][slot_index]) / self.status['num_pulls'][slot_index]
        N = self.status['num_pulls'].sum()
        nj = self.status['num_pulls'][slot_index]
        self.status['upper_bound'][slot_index] = np.sqrt(2 * np.log(N) / nj)

        self.status['p_estimate'][slot_index] = self.status['mean'][slot_index] + self.status['upper_bound'][slot_index]


class BayesianAgent(BaseAgent):
    """
    Implement the Thompson Sampling Theory
    """
    def __init__(self, num_slots):
        super().__init__(num_slots)
        self.status = {
            'alpha': np.array([1.0] * self.num_slots),
            'beta': np.array([1.0] * self.num_slots),
            'num_pulls': np.array([0] * self.num_slots)
        }
        self.status.update({'p_estimate': beta.stats(self.status['alpha'],
                                                     self.status['beta'],
                                                     moments='m'
                                                     )})

    def update_status(self, outcome, slot_index):
        self.status['alpha'][slot_index] += outcome
        self.status['beta'][slot_index] += 1 - outcome
        self.status['num_pulls'][slot_index] += 1
        self.status['p_estimate'][slot_index] = beta.stats(self.status['alpha'][slot_index],
                                                           self.status['beta'][slot_index],
                                                           moments='m'
                                                           )

    def sample(self, slot_index):
        a = self.status['alpha'][slot_index]
        b = self.status['beta'][slot_index]
        return np.random.beta(a, b)

    def action(self):
        return np.argmax([self.sample(j) for j in range(self.num_slots)])


class BayesianAgentGaussianReward(BayesianAgent):
    """
    Implement the Thompson Sapling Theory
    """
    def __init__(self, num_slots):
        super().__init__(num_slots)
        self.status = {
            'mean': np.array([1.0] * self.num_slots),
            'lambda': np.array([1.0] * self.num_slots),
            'num_pulls': np.array([0] * self.num_slots)
        }
        self.status.update({'p_estimate': self.status['mean']})

    def update_status(self, outcome, slot_index):
        m0 = self.status['mean'][slot_index]
        lambda0 = self.status['lambda'][slot_index]
        self.status['lambda'][slot_index] += 1
        self.status['mean'][slot_index] = (outcome + lambda0 * m0) / self.status['lambda'][slot_index]
        self.status['num_pulls'][slot_index] += 1
        self.status['p_estimate'][slot_index] = self.status['mean'][slot_index]

    def sample(self, slot_index):
        mean = self.status['mean'][slot_index]
        sigma = np.sqrt(1./self.status['lambda'][slot_index])
        return np.random.normal(mean, sigma)
