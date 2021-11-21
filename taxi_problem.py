import random
import time
from typing import Optional

import gym
from gym import Env

from QTable import GreedyQTable, TrainedQTable


def clear_console():
    """
    Clears the console's display.

    :return: None
    """
    import os
    clear_command: str = "cls"
    if os.name != "ns":
        clear_command = "clear"
    os.system(clear_command)


def solve_taxi_problem() -> GreedyQTable:
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
    initial_epsilon = 1  # Determines likelihood that the AI will explore over exploit for a given step
    max_epsilon = 1  # Maximum value for epsilon | Always exploring
    min_epsilon = 0.01  # Minimum value for epsilon | Mostly exploiting
    decay = 0.01  # Decay factor in epsilon between episode runs

    train_episodes = 2000
    max_steps = 100

    # STEP 1: Initializing the Q-table
    # q = np.zeros([env.observation_space.n, env.action_space.n])
    q = GreedyQTable(env.observation_space.n, env.action_space.n, alpha, discount_factor,
                     initial_epsilon, max_epsilon, min_epsilon, decay, env.action_space)

    # Creating lists to keep track of reward and epsilon values
    training_rewards = []
    epsilons = []

    print("Finding ideal solution to the Taxi game")

    for episode in range(train_episodes):
        # Resetting the environment each time as per requirement
        state = env.reset()
        q.reset_state_field(state)
        # Starting the tracker for the rewards
        total_training_rewards = 0

        for step in range(max_steps):
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            # STEP 2: Choose an action
            action = q.get_next_action(exp_exp_tradeoff)

            # STEP 3 & 4: perform the action and get the reward
            # Take the action and getting the reward and outcome state
            new_state, reward, done, _ = env.step(action)

            # STEP 5: Update the Q-table
            q.update_table(new_state, reward)

            # Increasing the total reward and updating the state
            total_training_rewards += reward

            # Check if end of the episode
            if done:
                # print(f"Total reward for episode {episode}: {total_training_rewards}")
                break

        # Cutting down on exploration by reducing the epsilon
        q.update_epsilon(episode)

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(q.epsilon)

    env.close()
    print(f"Training score over time: {str(sum(training_rewards)/train_episodes)}\n")
    return q


def evaluate_agent(q: TrainedQTable, episodes: int = 100):
    if type(q) is not TrainedQTable:
        raise TypeError

    if episodes <= 0:
        raise ValueError

    total_epochs = 0
    total_penalties = 0
    total_rewards = 0

    env: Env = gym.make('Taxi-v3')

    for _ in range(episodes):
        q.reset_state_field(env.reset())

        epochs = 0
        penalties = 0
        rewards = 0

        done = False
        while not done:
            action = q.get_next_action()
            state, reward, done, _ = env.step(action)

            if reward == -10:
                penalties += 1
            epochs += 1
            rewards += reward

            q.state = state
        total_penalties += penalties
        total_epochs += epochs
        total_rewards += rewards

    print(f"Performance results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average rewards per episode: {total_rewards / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")


def run_taxi_problem(q: TrainedQTable = None):
    """
    Runs a game of Taxi using a given Q-Table

    :param q: The Q-Table or None to run a game of random actions
    :return: None
    """
    # STEP  1: Setup the Gym environment
    env: Env = gym.make("Taxi-v3")

    # Get initial state for the environment
    state = env.reset()
    if q is not None:
        q.reset_state_field(state)

    # Render the initial state
    env.render()

    # Starting the tracker for the rewards
    total_training_rewards = 0
    done = False
    while not done:
        # STEP 2: Choose the ideal action to take
        if q is not None:
            action = q.get_next_action()
        else:
            action = env.action_space.sample()  # Take a random action

        # STEP 3 & 4: perform the action and get the reward

        # Take the action and getting the reward and outcome state
        new_state, reward, done, _ = env.step(action)

        # Increasing the total reward and updating the state
        total_training_rewards += reward

        if q is not None:
            q.state = new_state

        # Render this step
        clear_console()
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
    q: Optional[TrainedQTable] = None
    try:
        q = TrainedQTable.from_csv("greedy_qtable.csv")
    except IOError:
        print("No existing trained QTable")

    # Run a random or ideal game
    run_taxi_problem(q)

    if q is None:
        # Find an ideal solution
        q = solve_taxi_problem().to_trained_qtable()
        # Run an ideal game
        run_taxi_problem(q)
        q.save_to_csv("greedy_qtable.csv")

    evaluate_agent(q)
    print("")
    print("*** Q Table ***")
    print(q)


# Runs if the script is called directly
if __name__ == '__main__':
    main()
