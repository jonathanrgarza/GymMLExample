import numpy as np
from numpy import ndarray


class QTable:
    def __init__(self, num_states: int, num_actions: int, learning_rate: float, discount_factor: float,
                 q: np.ndarray = None):
        self.num_states: int = num_states
        self.num_actions: int = num_actions
        self.learning_rate: float = learning_rate  # alpha
        self.discount_factor: float = discount_factor  # gamma
        self.state = 0
        self.action = 0
        if q is ndarray:
            self.q = q.copy()
        else:
            self.q: ndarray = np.zeros([num_states, num_actions])

    @staticmethod
    def from_csv(path: str, learning_rate: float, discount_factor: float, *args, **kwargs):
        q: ndarray = np.loadtxt(path, delimiter=",")
        if q is None:
            raise IOError
        num_states, num_actions = q.shape
        return QTable(num_states, num_actions, learning_rate, discount_factor, q)

    def reset_state_field(self, state) -> None:
        self.state = state

    def get_next_action(self, episode: int):
        state = self.state
        self.action = np.argmax(self.q[state, :] + np.random.rand(1, self.num_actions) * (1. / (episode + 1)))
        return self.action

    def update_table(self, new_state, reward: int) -> None:
        state = self.state
        action = self.action
        self.q[state, action] = self.q[state, action] + self.learning_rate * (reward + self.discount_factor *
                                                                              np.max(self.q[new_state, :]) -
                                                                              self.q[state, action])
        self.state = new_state

    def save_to_csv(self, path: str):
        if path is None:
            raise TypeError
        # noinspection PyTypeChecker
        return np.savetxt(path, self.q, delimiter=",")

    def to_trained_qtable(self):
        return TrainedQTable(self.q)

    def __str__(self):
        import pandas as pd
        # return f"Q-Table:\n{np.self.q}"
        return pd.DataFrame(self.q).to_string()


class GreedyQTable(QTable):
    def __init__(self, num_states: int, num_actions: int, learning_rate: float, discount_factor: float, epsilon: float,
                 max_epsilon: float, min_epsilon: float, decay: float, action_space,
                 q: np.ndarray = None):
        super().__init__(num_states, num_actions, learning_rate, discount_factor, q)
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.action_space = action_space

    @staticmethod
    def from_csv(path: str, learning_rate: float, discount_factor: float, epsilon: float = 1.0,
                 max_epsilon: float = 1.0, min_epsilon: float = 0.1, decay: float = 0.1,
                 action_space=None, *args, **kwargs):
        q: ndarray = np.loadtxt(path, delimiter=",")
        if q is None:
            raise IOError
        num_states, num_actions = q.shape
        return GreedyQTable(num_states, num_actions, learning_rate, discount_factor, epsilon, max_epsilon, min_epsilon,
                            decay, action_space, q)

    def get_next_action(self, value: float):
        # STEP 2: First option for choosing the action - explore
        if value <= self.epsilon:
            self.action = self.action_space.sample()
        # STEP 2: Second option for choosing the action - exploit
        else:
            self.get_ideal_action()
        return self.action

    def get_ideal_action(self):
        self.action = np.argmax(self.q[self.state, :])
        return self.action

    def update_epsilon(self, episode: int):
        # Reducing epsilon value will reduce the number of exploration actions
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * episode)


class TrainedQTable:
    def __init__(self, q: np.ndarray):
        if type(q) is not np.ndarray:
            raise ValueError

        self.q = q.copy()
        self.state = 0
        self.action = 0

    @staticmethod
    def from_csv(path: str):
        q: ndarray = np.loadtxt(path, delimiter=",")
        if q is None:
            raise IOError
        return TrainedQTable(q)

    def reset_state_field(self, state) -> None:
        self.state = state

    def get_next_action(self):
        self.action = np.argmax(self.q[self.state, :])
        return self.action

    def save_to_csv(self, path: str):
        if path is None:
            raise TypeError
        # noinspection PyTypeChecker
        return np.savetxt(path, self.q, delimiter=",")

    def __str__(self):
        import pandas as pd
        # return f"Q-Table:\n{np.self.q}"
        return pd.DataFrame(self.q).to_string()
