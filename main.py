# This is a sample Python script with Gym.
import gym
import numpy as np


def run_minimum_gym():
    """
        Bare minimum example of getting gym running.
        This will run an instance of the CartPole-v0 environment of 1000 time steps,
        rendering the environment at each step.
    """

    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    env.close()


def run_better_gym():
    """
        A more complete example of getting gym running.
        This will run an instance of the CartPole-v0 environment of 1000 time steps,
        rendering the environment at each step.
    """
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} time steps".format(t+1))
                break
    env.close()


def run_q_learning_gym():
    import time

    # 1. Load Env. and Q-take structure
    env = gym.make('FrozenLake8x8-v1')
    q = np.zeros([env.observation_space.n, env.action_space.n])
    # env.observation_space.n, env.action_space.n gives number of states and action in env loaded

    # 2. Parameters of Q-Learning
    eta = .628  # discount factor ?
    gma = .9  # learning rate ?
    episodes = 5000
    rev_list = []  # rewards per episode calculate

    # 3. Q-learning algorithm
    for i in range(episodes):
        print()
        print("New Episode")
        # Reset environment
        s = env.reset()
        r_all = 0
        done = False
        j = 0
        # The Q-Table learning algorithm
        while not done:
            env.render()

            j += 1
            # Choose action from Q table
            a = np.argmax(q[s, :] + np.random.rand(1, env.action_space.n) * (1./(i + 1)))
            # Get new state & reward from environment
            s1, reward, done, _ = env.step(a)
            # Update Q-Table with new knowledge
            q[s, a] = q[s, a] + eta * (reward + gma * np.max(q[s1, :]) - q[s, a])
            r_all += reward
            s = s1

            # Wait a bit before the next frame
            time.sleep(0.001)
        rev_list.append(r_all)
        env.render()

    env.close()
    print()
    print("Reward Sum on all episodes: " + str(sum(rev_list) / episodes))
    print("Final Values Q-Table")
    print(q)


def show_gym_envs():
    """
    Shows all the gym environments currently installed
    """
    from gym import envs
    print(envs.registry.all())


# Runs if the script is called directly
if __name__ == '__main__':
    # show_gym_envs()
    # run_better_gym()
    run_q_learning_gym()
