import gym
import numpy as np
import random
from collections import defaultdict

env = gym.make('Blackjack-v0')

"""
MC-control:
   - 根据旧的策略， greedy改进得到new policy
   - 在游戏中generate new policy
   - 根据new policy更新Q表

关于Q表：利用字典储存，Q[state][action]:
   - state为一三元组，分别为(当前点数，庄家明牌点数，是否有ace(True/False))
   - action为一二元组，是该状态下再执行之后的策略的动作价值，分别是(动作0的价值，动作1的价值)

在python中利用defaultdict函数得到字典
"""

#不断迭代选择动作和更新Q表， 实现MC-control，与pro1有点像的函数结构
#num为循环次数（episode数量），epsilonde的线性衰减系数:eps_dec， 最小的epsilon值：eps_min
def MC_control(env, num, alpha, gamma = 1, initial_epsilon = 1, eps_dec=0.8, eps_min=0.01):
    #刚开始时，为initial_epsilon,在之后的循环中衰减
    epsilon = initial_epsilon
    #初始化Q表（字典）
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    #循环
    for i in range(1, num+1):
        #每一次循环线性衰减，但不可以小于eps_min
        epsilon = max(epsilon*eps_dec, eps_min)
        #基于Q表执行动作，得到下一个状态数组，动作数组和汇报数组（全部循环完毕才更新Q表，与TD不同）
        next_states, actions, rewards, _ = generate_based_Q(env, Q, epsilon, env.action_space.n)
        #根据动作执行的反馈更新Q表
        Q = new_Q(next_states, actions, rewards, Q, alpha, gamma)

    #根据Q表得到policy
    policy = dict((k, np.argmax(v)) for k, v in Q.items())

    return policy, Q

#greedy策略得到动作的分布概率，输出和动作空间相同大小的矩阵（元素为对应动作的概率）
def greedy_select(Q, epsilon, nA):
    policy_pro = np.ones(nA)*epsilon / nA
    select_action = np.argmax(Q)
    policy_pro[select_action] = 1 - epsilon + (epsilon / nA)
    return policy_pro

#根据Q表greedy选择动作得到新策略和新策略的state，action， reward
def generate_based_Q(env, Q, epsilon, nA):
    states, actions, rewards = [], [], []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=greedy_select(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if done:
            break

    return states, actions, rewards, reward

#更新Q表
def new_Q(states, actions, rewards, Q, alpha, gamma):
    discounts = np.array([gamma**i for i in range(len(states)+1)])
    for i, state in enumerate(states):
        #新估计-旧Q
        Q[state][actions[i]] = Q[state][actions[i]] + alpha*sum(rewards[i:]*discounts[:-(1+i)]- Q[state][actions[i]])
    return Q

"""
TD control
主要差别在每一步都要更新Q表，MC循环完整个episode更新
"""
def TD_control(env, num, alpha, gamma=1.0,initial_epsilon = 1, eps_dec=0.8, eps_min=0.01):
    #初始化Q
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    epsilon = initial_epsilon
    #循环每个episode
    for i in range(1, num+1):
        score = 0
        state = env.reset()
        epsilon = max(epsilon*eps_dec, eps_min)
        #greedy选择动作，先选择动作，执行一步更新一步Q表，此处新定义一下greedy函数，MC所用的返回结果无法使用
        action = epsilon_greedy(Q, state, epsilon, env.action_space.n)
        #在同一个episode中，不断执行action更新Q表
        while True:
            next_state, reward, done, _ = env.step(action)
            score += reward
            if not done:
                #greedy选择下一动作（根据next_state）
                next_action = epsilon_greedy(Q, next_state, epsilon, env.action_space.n)
                #更新Q表
                Q[state][action] = new_Q_TD(alpha, gamma, Q, state, action, reward, next_state, next_action)
                #更新状态和动作
                state = next_state
                action = next_action
            if done:
                Q[state][action] = new_Q_TD(alpha, gamma, Q, state, action, reward)
                break
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q

def new_Q_TD(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):

    current = Q[state][action]
    Q_next = Q[next_state][next_action] if next_state is not None else 0
    #得到target，用于计算新的value
    target = reward + (gamma * Q_next)
    #新的value公式
    new_value = current + (alpha * (target - current))
    return new_value

def epsilon_greedy(Q, state, epsilon, nA):
    select_action = np.argmax(Q[state])
    policy_pro = np.ones(nA)*epsilon / nA
    policy_pro[select_action] = 1-epsilon + epsilon / nA
    return np.random.choice(np.arange(env.action_space.n),p=policy_pro)

"""
基于得到的Q表运行多次函数（利用MC中使用的那一个函数即可），测试函数
"""
#运行函数，测试所求策略下的胜率
def check_policy(env, Q, episodes, epsilon, nA):
    nums = 0
    for i in range(episodes):
        _, _, _, reward = generate_based_Q(env, Q, epsilon, nA)
        #计算胜率时，认为‘不输’，即reward不等于-1即为胜利
        if reward == -1:
            nums += 1
    return 1 - nums / episodes


"""
MC训练和测试结果：
    -5000次episode训练时，胜率为0.4681
    -50000次episode训练时，胜率为0.4848
    -500000次episode训练时，胜率为0.4936
"""
MC_policy, MC_Q = MC_control(env, 50000, 0.0015)
#执行训练得到的策略时，epsilon取0.1
#测试5000次，测试时alpha取0.0015
MCacc = check_policy(env, MC_Q, episodes=5000, epsilon=0.1, nA= env.action_space.n)
#print("MC得到的最终Q表为：",MC_Q)
#print("MC得到的最终策略为：",MC_policy)
print("MC最终胜率(平局或胜利)为：", MCacc)


"""
MC训练和测试结果：
     -5000次episode训练时，胜率为0.466
     -50000次episode训练时，胜率为0.498
     -500000次episode训练时，胜率为0.4946
"""
TD_policy, TD_Q = TD_control(env, 50000, 0.009)
#执行训练得到的策略时，epsilon取0.1，#测试5000次
TDacc = check_policy(env, TD_Q, episodes=5000, epsilon=0.1, nA= env.action_space.n)
#print("TD得到的最终Q表为：", TD_Q)
#print("TD得到的最终策略为：",TD_policy)
print("TD最终胜率(平局或胜利)为：", TDacc)
