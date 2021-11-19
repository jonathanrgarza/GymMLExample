import numpy as np

from NeuralNetwork import NeuralNetwork


def compute_dot_products():
    input_vector = [1.72, 1.23]
    weights_1 = [1.26, 0]
    weights_2 = [2.17, 0.32]

    # Computing the dot product of input_vector and weights_1
    first_indexes_mult = input_vector[0] * weights_1[0]
    second_indexes_mult = input_vector[1] * weights_1[1]
    dot_product_1 = first_indexes_mult + second_indexes_mult

    print(f"The dot product is: {dot_product_1}")

    # NumPy way of computing dot product
    dot_product_1 = np.dot(input_vector, weights_1)
    print(f"The dot product is: {dot_product_1}")

    dot_product_2 = np.dot(input_vector, weights_2)
    print(f"The dot product is: {dot_product_2}")

    # Higher dot product means more similar to vector


def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid value for a given input.
    :param x: The input value.
    :return: The sigmoid value (A value between 0 and 1).
    """
    return 1 / (1 + np.exp(-x))


def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2


def example1():
    # Wrapping the vectors in NumPy arrays
    # input_vector = np.array([1.66, 1.56])
    input_vector = np.array([2, 1.5])
    weights_1 = np.array([1.45, -0.66])
    bias = np.array([0.0])

    prediction = make_prediction(input_vector, weights_1, bias)
    # print(f"The prediction result is: {prediction}")

    # Computer error
    target = 0
    # Mean Squared Error (MSE)
    mse = np.square(prediction - target)
    print(f"Prediction: {prediction}; Error: {mse}")
    # One implication of multiplying the difference by itself is that bigger errors have
    # an even larger impact, and smaller errors keep getting smaller as they decrease

    # Determines how we should decrease the error
    derivative = 2 * (prediction - target)
    print(f"The derivative is {derivative}")

    # Updating the weights
    weights_1 = weights_1 - derivative

    prediction = make_prediction(input_vector, weights_1, bias)

    error = (prediction - target) ** 2
    print(f"Prediction: {prediction}; Error: {error}")

    learning_rate = 0.1
    neural_network = NeuralNetwork(learning_rate)
    prediction = neural_network.predict(input_vector)
    print(f"The prediction result using NeuralNetwork is: {prediction}")


def main():
    import matplotlib.pyplot as plt

    input_vectors = np.array(
        [
            [3, 1.5],
            [2, 1],
            [4, 1.5],
            [3.5, 0.5],
            [2, 0.5],
            [5.5, 1],
            [1, 1]
        ])
    targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    learning_rate = 0.1

    neural_network = NeuralNetwork(learning_rate)
    training_error = neural_network.train(input_vectors, targets, 10000)

    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    # plt.savefig("cumulative_error.png")
    plt.show()


# Runs if the script is called directly
if __name__ == '__main__':
    main()
