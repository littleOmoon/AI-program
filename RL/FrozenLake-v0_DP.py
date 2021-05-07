from gym import envs
import gym
import numpy as np

#建立冰湖环境
env = gym.make('FrozenLake-v0')

# 已有策略后，执行策略的函数；返回总收益
def execute_policy(env, policy):
    #每次执行前初始化
    total_reward = 0
    observation = env.reset()
    while True:
        #根据策略获得动作
        action = np.random.choice(env.nA, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

#策略评估
def evaluate_policy(env, policy, gamma, threshold):
    #初始化一个和状态表格相同大小的状态价值表格
    value_table = np.zeros(env.nS)
    while True:
        delta = 0.0
        #  对于每个状态的每个动作进行贪心选择
        for state in range(env.nS):
            Q = []
            vs = 0
            for action in range(env.nA):
                #trans_prob是状态转移到下一个状态的概率（环境的参数）
                q = 0
                for i in range(np.shape(env.P[state][action])[0]):
                    trans_prob, next_state, reward, _ = env.P[state][action][i]
                    q =q + trans_prob*(reward + gamma*value_table[next_state])
                Q.append(q)

            vs = sum(policy[state]*Q)
            #上一个delta和下一个delta取最大值
            delta = max(delta , abs(vs-value_table[state]))
            value_table[state] = vs
        if delta < threshold:
            break
    return value_table


# 根据状态评估函数获得的价值表格改进（策略改进）
def improve_policy(env,  policy, gamma, threshold):
    value_table = evaluate_policy(env, policy, gamma, threshold)
    optimal = True
    # 利用bellman方程计算q
    for state in range(env.nS):
        Q_table = np.zeros(env.nA)
        for action in range(env.nA):
            for i in range(np.shape(env.P[state][action])[0]):
                trans_prob, next_state, reward, _ = env.P[state][action][i]
                Q_table[action] = Q_table[action] + trans_prob*(reward + gamma*value_table[next_state])

        a = np.argmax(Q_table)
        #用varepsilon-greedy改进，增加exploration，在该问题上使达到收敛的迭代次数变得不定，对问题解决没有帮助
        # var=0.1
        # a_max = np.argmax(Q_table)
        # ap = np.zeros(env.nA)
        # ap[:] = var / (env.nA - 1)
        # ap[a_max] = 1-var
        # a = np.random.choice(a=env.nA, p=ap)
        if policy[state][a] != 1.:
            optimal = False
            policy[state] = 0.
            policy[state][a] = 1.
    return optimal


# 策略迭代:不断评估和改进策略
def iterate_policy(env, gamma, threshold):
    policy = np.ones((env.nS, env.nA)) / env.nA
    for i in range(1000):
        value_table = evaluate_policy(env, policy, gamma, threshold)
        if improve_policy(env, policy, gamma, threshold):
            print("达到收敛的迭代次数： %d " %(i+1))
            break
    return policy, value_table


def check_policy(policy, episodes):
    successed_nums = 0
    for i in range(episodes):
        one_episode_return = execute_policy(env, policy)
        if one_episode_return == 1.0:
            successed_nums += 1
    return  successed_nums / episodes

gamma = 1
threshold = 1e-3
optimal_policy, optimal_value_tabel = iterate_policy(env,gamma,threshold)
print("最终策略: ", optimal_policy, sep='\n')
print("状态价值表: ", optimal_value_tabel, sep='\n')

acc = check_policy(optimal_policy, episodes=500)
print("成功率: ", acc)
