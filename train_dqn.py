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

env = gym.make('CartPole-v0')
#env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity=pow(10,5)):
        # a list containing four elements tuple ( s_t, a, r, s_{t+1} )
        self.buffer = []
        self.capacity = capacity
        self.cursor = 0
        self.batch_size = 64

    def add(self, quad_tuple):
        if(len(self.buffer)<self.capacity):        
            self.buffer.append(quad_tuple)
        else:
            self.buffer[self.cursor] = quad_tuple
            self.cursor = (self.cursor + 1)%self.capacity

    def get_size(self):
        return len(self.buffer)

    def is_full(self):
        return (self.get_size() == self.capacity)
 
    def sample(self):
        return self.buffer[self._uniform_sampling()]

    def _uniform_sampling(self):
        return np.random.choice(np.arange(len(self.buffer)))      
    
    def _prioritized_sampling(self):
        raise NotImplementedError
        

class QNetwork(nn.Module):
    def __init__(self, s_size=4, h_size=128, a_size=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

        self.action_size = a_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def act(self, state, eps=0.01):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = self.forward(state).cpu()
        p = np.random.random()
        if p < eps:
            return np.random.choice(range(self.action_size))
        else:
            return np.argmax(value.detach().numpy())


dqn = QNetwork().to(device)
optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(capacity=10000)

state = env.reset()
import time

total_reward = 0
for t in range(1000):
    action = dqn.act(state, eps=0)
    env.render()
    state, reward, done, info = env.step(action)

    time.sleep(0.03)
    total_reward += (reward)
    if done:
        break 

print("before training: {}".format(total_reward))



from tqdm import tqdm
gamma = 0.98
batch_size = 64

# collecting data
state = env.reset()

while True:
    action = dqn.act(state)
    next_state, reward, done, info = env.step(action)

    replay_buffer.add((state, action, reward, next_state))
    if replay_buffer.is_full():
        break

    state = next_state
    if done:
        state = env.reset() 


print("Collecting experiences: {}".format(replay_buffer.get_size()))

# after collecting enough experiences
# starts to train
num_episodes = 500
for e in range(num_episodes):
    
    state = env.reset()

    # fixing the episode length
    for i in range(100):
        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)

        replay_buffer.add((state, action, reward, next_state))
        state = next_state

        #total_reward += (reward)
        if done:
            break 
    
    optimizer.zero_grad()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    batch_size = 64
    for i in range(batch_size):
        state, action, reward, next_state = replay_buffer.sample()
        y = reward + gamma*torch.max(dqn(torch.from_numpy(next_state).to(device)))
        q_y = dqn(torch.from_numpy(state).to(device))[action]
        total_loss += criterion(y, q_y)

    total_loss = total_loss/batch_size
    total_loss.backward()
    optimizer.step()


total_reward = 0
state = env.reset()
for t in range(1000):
    action = dqn.act(state,eps=0)
    env.render()
    state, reward, done, info = env.step(action)

    time.sleep(0.03)
    total_reward += (reward)
    if done:
        break 

print("after training: {}".format(total_reward))


env.close()

