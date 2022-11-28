import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

env_name = "CartPole-v0"

env = gym.make(env_name)

env.action_space.seed(42)

print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)



state, _ = env.reset()
import time

total_reward = 0
for t in range(1000):
    action, _ = policy.act(state)
    env.render()
    state, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.03)
    total_reward += (reward)
    if terminated or truncated:
        break 

print("before training: {}".format(total_reward))


#exit()

# how long can you keep
return_list = []
from tqdm import tqdm
gamma = 1.0

batch_size = 2

# how many step to perform
for episode_num in tqdm(range(200)):
    
    
    # do we need multiple batch
    for q in range(batch_size):
        optimizer.zero_grad()
        episode_loss = 0    


        cum_reward = 0 
        state, _ = env.reset()
        #gamma_st = 1
        reward_list = []        
        lgprob_list = []
    
        for t in range(200):
            action, log_prob = policy.act(state)
    #        env.render()
            state, reward, terminated, truncated, info = env.step(action)

            reward_list.append(reward)
            lgprob_list.append(log_prob)
            cum_reward = cum_reward + reward
            #gamma_st = gamma_st * gamma
            if terminated or truncated:
                break
                    

            # aggregating reward to go 
        for i in range(len(reward_list)):
            episode_loss = episode_loss - lgprob_list[i]*sum(reward_list[i:])

    # a single batch
    return_list.append(cum_reward)
        #print(episode_loss)

    episode_loss /= batch_size
    episode_loss.backward()
    # I miss this step but still improve...?    
    optimizer.step()


total_reward = 0
state,_ = env.reset()
for t in range(1000):
    action, _ = policy.act(state)
    env.render()
    state, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.03)
    total_reward += (reward)
    if terminated or truncated:
        break 

print("after training: {}".format(total_reward))

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig('./return.png')

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.savefig('./average_return.png')

env.close()

