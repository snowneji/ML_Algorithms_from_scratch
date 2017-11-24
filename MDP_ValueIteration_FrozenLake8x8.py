"""
You are being asked to explore Markov Decision Processes (MDPs):

Come up with two interesting MDPs.

Explain why they are interesting.
They don't need to be overly complicated or directly grounded in a real situation,
but it will be worthwhile if your MDPs are inspired by some process you are interested
 in or are familiar with.
 It's ok to keep it somewhat simple.
 For the purposes of this assignment, though,
 make sure one has a "small" number of states,
 and the other has a "large" number of states.




Solve each MDP using value iteration
as well as policy iteration.

How many iterations does it take to converge?
Which one converges faster? Why?
Do they converge to the same answer?
How did the number of states affect things, if at all?


Now pick your favorite reinforcement learning algorithm and use it
 to solve the two MDPs.
 How does it perform,
 especially in comparison to the cases above
  where you knew the model, rewards, and so on?
  What exploration strategies did you choose? Did some work better than others?
"""


"""
Solving FrozenLake8x8 environment using Value-Itertion.
Reference: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
"""
import numpy as np
import gym
from gym import wrappers
import time
import pandas as pd



def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.unwrapped.nS)
    for s in range(env.unwrapped.nS): # n state
        q_sa = np.zeros(env.action_space.n) # n action
        for a in range(env.action_space.n):
            for next_sr in env.unwrapped.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.unwrapped.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.unwrapped.nS): # n state
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.unwrapped.P[s][a]]) for a in range(env.unwrapped.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps): # when the v doesn't imporve, MDP is converged
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v


if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v0'
    gamma = 1
    N_ITER = 10

    AVG_ROUND = []
    AVG_TIME = []
    AVG_SCORE = []

    for i in range(N_ITER):
        start = time.time()
        env = gym.make(env_name)
        res = value_iteration(env, gamma)#[0] # get the optimal value
        optimal_v = res[0]
        policy = extract_policy(optimal_v, gamma) # get the policy
        policy_score = evaluate_policy(env, policy, gamma, n=5000)

        AVG_TIME.append(time.time()-start)
        AVG_ROUND.append(res[1])
        AVG_SCORE.append(policy_score)





