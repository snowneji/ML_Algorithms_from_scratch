import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
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


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.unwrapped.nS)
    for s in range(env.unwrapped.nS):
        q_sa = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.unwrapped.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.unwrapped.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.unwrapped.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.unwrapped.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.unwrapped.nA, size=(env.unwrapped.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)

        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return (policy,(i+1))


if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    N_ITER = 50

    AVG_ROUND = []
    AVG_TIME = []
    AVG_SCORE = []

    for i in range(N_ITER):
        start = time.time()
        env = gym.make(env_name)
        res = policy_iteration(env, gamma = 1.0)
        optimal_policy = res[0]
        scores = evaluate_policy(env, optimal_policy, gamma = 1.0)

        AVG_TIME.append(time.time()-start)
        AVG_ROUND.append(res[1])
        AVG_SCORE.append(scores)


    print('Policy average time = ', np.mean(AVG_TIME))
    print('Policy average round = ', np.mean(AVG_ROUND))
    print('Policy average score = ', np.mean(AVG_SCORE))


    final_res = pd.DataFrame({'time':AVG_TIME,'round':AVG_ROUND,'score':AVG_SCORE})

    final_res.mean()


