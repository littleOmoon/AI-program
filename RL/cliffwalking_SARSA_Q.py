import gym
import time
import numpy as np

'''
Q-learning 和 SARSA 在编程实现时大致的结构相似，需要动作选择函数（greedy，训练时使用），根据Q表选择动作的函数（测试时使用）和Q表更新函数
'''
class QLearningAgent(object):
    # 定义该有的参数
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，输出动作
    def ActionSelect(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon): #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n) #有一定概率随机探索选取一个动作
        return action

    # 根据观察值，预测输出动作
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    #off-policy
    def Qlearn(self, obs, action, reward, next_obs, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward # 没有下一个状态
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :]) # Q-learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q) # 修正q


class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，输出动作（greedy，训练agent时使用）
    def ActionSelect(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon): #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n) #有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值（测试agent时使用）
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    # on-policy
    def Slearn(self, obs, action, reward, next_obs, next_action, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
    #对比：target_Q = reward + self.gamma * np.max(self.Q[next_obs, :]) # Q-learning
            target_Q = reward + self.gamma * self.Q[next_obs, next_action] # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q) # 修正q


'''
两个运行、训练agent的函数，仅有更新Q表时不同
'''
def Qrun_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.ActionSelect(obs) # 根据算法选择动作
        next_obs, reward, done, _ = env.step(action) # 与环境进行交互
        # 更新Q表（利用Q-learning算法对agent进行训练）
        agent.Qlearn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps

#和Q-learning的类似，只有在更新Q表时不同
def Srun_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.ActionSelect(obs) # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action) # 与环境进行一个交互
        next_action = agent.ActionSelect(next_obs) # 根据算法选择一个动作
        # 更新Q表
        agent.Slearn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps

#测试函数
def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        #输出测试阶段每一步的选择
        time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward

# 使用gym创建悬崖环境
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
'''
创建两个agent分别训练，控制超参数相同，以便之后比较两个算法
'''
# 创建一个Q_learning 的agent实例，输入超参数
Qagent = QLearningAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.9,
    e_greed=0.1)
# 创建一个agent实例，输入超参数
Sagent = SarsaAgent(
    obs_n=env.observation_space.n,
    act_n=env.action_space.n,
    learning_rate=0.1,
    gamma=0.9,
    e_greed=0.1)


# 训练500个episode，打印每个episode的分数
for episode in range(500):
    ep_reward, ep_steps = Qrun_episode(env, Qagent, False)
    #print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 全部训练结束，查看算法效果
print('Q-learning results：')
test_reward = test_episode(env, Qagent)
print('Q learning test reward = %.1f' % (test_reward))


# 训练500个episode，打印每个episode的分数
for episode in range(500):
    ep_reward, ep_steps = Srun_episode(env, Sagent, False)
    #print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 全部训练结束，查看算法效果
print('SARSA results：')
test_reward = test_episode(env, Sagent)
print('SARSA test reward = %.1f' % (test_reward))

'''
从打印出的测试结果可以看出：
当所有参数都一样的时候
Q-learning的结果最终的reward更大，会更靠近悬崖行走
sarsa的结果最终的reward更小，会更远离悬崖行走（safer）
对比：target_Q = reward + self.gamma * np.max(self.Q[next_obs, :]) # Q-learning
     target_Q = reward + self.gamma * self.Q[next_obs, next_action] # Sarsa
     
     Q-learning选择的是之后的一个状态最大action的价值，而sarsa则是选择之后的那一个action的价值，从这里可以看出，Q-learning最终会选择的策略会更加的’激进‘和’冒险‘

在测试过程中，sarsa训练出来的模型有时会出现卡在右上角不动的情况，暂时没有搞明白为什么
'''
