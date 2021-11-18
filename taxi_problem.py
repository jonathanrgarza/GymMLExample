import random

import gym
import numpy as np
from gym import Env


def solve_taxi_problem():
    env: Env = gym.make("Taxi-v3")
    env.reset()
    env.render()

    # Q-Learning Algorithm parameters
    alpha = 0.7  # learning rate | the factor that newer data is
    discount_factor = 0.618  # The factor that older data loses value
    epsilon = 1  # ?
    max_epsilon = 1  # ?
    min_epsilon = 0.01  # ?
    decay = 0.01  #

    train_episodes = 2000
    test_episodes = 100
    max_steps = 100

    # Initializing the Q-table
    q = np.zeros([env.observation_space.n, env.action_space.n])

    # Creating lists to keep track of reward and epsilon values
    training_rewards = []
    epsilons = []

    for episode in range(train_episodes):
        # Resetting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(100):
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
            new_state, reward, done, info = env.step(action)

            # STEP 5: Update the Q-table
            q[state, action] = q[state, action] + alpha * (reward + discount_factor *
                                                           np.max(q[new_state, :]) - q[state, action])

            # Increasing the total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Check if end of the episode
            if done:
                print(f"Total reward for episode {episode}: {total_training_rewards}")
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

    print(f"Training score over time: {str(sum(training_rewards)/train_episodes)}")


def main():
    solve_taxi_problem()


# Runs if the script is called directly
if __name__ == '__main__':
    main()
