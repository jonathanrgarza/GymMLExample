import numpy as np


class NeuralNetwork:
    """
    A neural network using just python and NumPy that can predict and train on simple data sets.
    """
    def __init__(self, learning_rate: float):
        self.weights = np.array([np.random.rand(), np.random.rand()])
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    # noinspection PyMethodMayBeStatic
    def _sigmoid(self, x: float) -> float:
        """
            Calculates the sigmoid value for a given input.
            :param x: The input value.
            :return: The sigmoid value (A value between 0 and 1).
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        sigmoid = self._sigmoid(x)
        return sigmoid * (1 - sigmoid)

    def predict(self, input_vector) -> float:
        """
        Performs a prediction based on the current state of the neural network
        :param input_vector: The input vector
        :return: The prediction value
        """
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_derivative(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias: float, derror_dweights: float):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instance (every 100 instances)
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors
