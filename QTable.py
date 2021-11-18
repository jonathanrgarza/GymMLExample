import numpy as np
from numpy import ndarray


class QTable:
    def __init__(self, num_states: int, num_actions: int, alpha: float, discount_factor: float):
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.q: ndarray = np.zeros([num_states, num_actions])
        self.alpha: float = alpha
        self.discount_factor: float = discount_factor
        self.state = 0
        self.action = 0

    def get_next_action(self, episode: int):
        state = self.state
        self.action = np.argmax(self.q[state, :] + np.random.rand(1, self.num_actions) * (1. / (episode + 1)))
        return self.action

    def update_table(self, new_state, reward):
        state = self.state
        action = self.action
        self.q[state, action] = self.q[state, action] + self.alpha * (reward + self.discount_factor *
                                                                      np.max(self.q[new_state, :]) -
                                                                      self.q[state, action])
        self.state = new_state
