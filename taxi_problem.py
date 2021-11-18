import random
import time

import gym
import numpy as np
from gym import Env
from numpy import ndarray


def solve_taxi_problem() -> ndarray:
    """
    Determines an ideal solution to the Taxi-v3 Gym environment
    using Q-Table learning

    :return: The Q-Table
    """
    # Setup the Gym environment
    env: Env = gym.make("Taxi-v3")
    # env.reset()
    # env.render()

    # Q-Learning Algorithm parameters
    alpha = 0.7  # learning rate | The factor that newly acquired information overrides old information
    discount_factor = 0.618  # The factor which determines the valuing of rewards received earlier higher than those
    # received later (reflecting a "good start")
    epsilon = 1  # Determines likelihood that the AI will explore over exploit for a given step
    max_epsilon = 1  # Maximum value for epsilon | Always exploring
    min_epsilon = 0.01  # Minimum value for epsilon | Mostly exploiting
    decay = 0.01  # Decay factor in epsilon between episode runs

    train_episodes = 2000
    max_steps = 100

    # STEP 1: Initializing the Q-table
    q: ndarray = np.zeros([env.observation_space.n, env.action_space.n])

    # Creating lists to keep track of reward and epsilon values
    training_rewards = []
    epsilons = []

    print("Finding ideal solution to the Taxi game")

    for episode in range(train_episodes):
        # Resetting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(max_steps):
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            # STEP 2: First option for choosing the initial action - explore
            if exp_exp_tradeoff <= epsilon:
                action = env.action_space.sample()
            # STEP 2: Second option for choosing the initial action - exploit
            else:
                action = np.argmax(q[state, :])

            # STEP 3 & 4: perform the action and get the reward

            # Take the action and getting the reward and outcome state
            new_state, reward, done, _ = env.step(action)

            # STEP 5: Update the Q-table
            q[state, action] = q[state, action] + alpha * (reward + discount_factor *
                                                           np.max(q[new_state, :]) - q[state, action])

            # Increasing the total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Check if end of the episode
            if done:
                # print(f"Total reward for episode {episode}: {total_training_rewards}")
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

    env.close()
    print(f"Training score over time: {str(sum(training_rewards)/train_episodes)}\n")
    return q


def run_taxi_problem(q: ndarray = None):
    """
    Runs a game of Taxi using a given Q-Table
    :param q: The Q-Table or None to run a game of random actions
    :return: None
    """
    # STEP  1: Setup the Gym environment
    env: Env = gym.make("Taxi-v3")

    # Get initial state for the environment
    state = env.reset()
    # Render the initial state
    env.render()

    # Starting the tracker for the rewards
    total_training_rewards = 0
    done = False
    while not done:
        # STEP 2: Choose the ideal action to take
        if q is not None:
            action = np.argmax(q[state, :])
        else:
            action = env.action_space.sample()  # Take a random action

        # STEP 3 & 4: perform the action and get the reward

        # Take the action and getting the reward and outcome state
        new_state, reward, done, _ = env.step(action)

        # Increasing the total reward and updating the state
        total_training_rewards += reward
        state = new_state

        # Render this step
        env.render()
        print(f"Reward: {reward}\n")

        # Wait a bit before the next frame
        if q is not None:
            time.sleep(0.7)
        else:
            time.sleep(0.001)

    print("Taxi Game Complete")
    print(f"Score: {total_training_rewards}\n")
    env.close()


def main():
    # Run a random game
    run_taxi_problem()
    # Find an ideal solution
    q = solve_taxi_problem()
    # Run an ideal game
    run_taxi_problem(q)


# Runs if the script is called directly
if __name__ == '__main__':
    main()
