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

    def reset_state_field(self, state) -> None:
        self.state = state

    def get_next_action(self, episode: int):
        state = self.state
        self.action = np.argmax(self.q[state, :] + np.random.rand(1, self.num_actions) * (1. / (episode + 1)))
        return self.action

    def update_table(self, new_state, reward: int) -> None:
        state = self.state
        action = self.action
        self.q[state, action] = self.q[state, action] + self.alpha * (reward + self.discount_factor *
                                                                      np.max(self.q[new_state, :]) -
                                                                      self.q[state, action])
        self.state = new_state

    def __str__(self):
        return f"Q-Table:\n{self.q}"


class GreedyQTable(QTable):
    def __init__(self, num_states: int, num_actions: int, alpha: float, discount_factor: float, epsilon: float, 
                 max_epsilon: float, min_epsilon: float, decay: float, action_space):
        super().__init__(num_states, num_actions, alpha, discount_factor)
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.action_space = action_space

    def get_next_action(self, value: float):
        # STEP 2: First option for choosing the action - explore
        if value <= self.epsilon:
            self.action = self.action_space.sample()
        # STEP 2: Second option for choosing the action - exploit
        else:
            self.action = np.argmax(self.q[self.state, :])
        return self.action

    def get_ideal_action(self):
        self.action = np.argmax(self.q[self.state, :])
        return self.action

    def update_epsilon(self, episode: int):
        # Reducing epsilon value will reduce the number of exploration actions
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * episode)
