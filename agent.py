import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import pickle
import sys

device = torch.device("cpu")


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.to(device)

    def forward(self, state):
        return self.model(state)


class Agent():
    
    def __init__(self, num_actions, eps_start=1.0, eps_end=0.05, eps_decay=0.996,
                            gamma=0.992, memory_capacity=20000, batch_size=64, alpha=1e-3, tau=1e-3):
        self.local_Q = Network().to(device)
        self.target_Q = Network().to(device)
        self.target_Q.load_state_dict(self.local_Q.state_dict())
        self.target_Q.eval()
        self.optimizer = optim.Adam(self.local_Q.parameters(), lr=alpha)
        self.loss = nn.SmoothL1Loss()
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(memory_capacity)
        self.indexes = np.arange(self.batch_size)
        self.scores = []
        self.episodes = []
        self.durations = []
        self.start = 1
    
    def store_experience(self, *args):
        self.replay_memory.push(args)

    def select_action(self, states):
        if np.random.random() > self.eps_start:
            with torch.no_grad():
                obs = torch.tensor(states, device=device, dtype=torch.float)
                action = torch.argmax(self.local_Q(obs)).item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def learn(self):
        ln = len(self.replay_memory.memory)
        if self.batch_size >= ln:# or ln < self.replay_memory.capacity:
            return
        
        state_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_memory.sample(self.batch_size)
        
        max_actions = torch.argmax(self.local_Q(next_state_batch), dim=1)
        prediction = self.local_Q(state_batch)[:,0]

        with torch.no_grad():
            evaluated = self.target_Q(next_state_batch)[:,0]
            evaluated = reward_batch + self.gamma * evaluated * done_batch

        self.optimizer.zero_grad()
        self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_Q.parameters(), self.local_Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
        
        self.eps_start = max(self.eps_end, self.eps_decay * self.eps_start)
        

    def save(self, filename):
        pickle_out = open(filename+".tt","wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()


def load_agent(filename):
    pickle_in = open(filename, mode="rb")
    agent = pickle.load(pickle_in)
    pickle_in.close()
    return agent


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        batch = list(zip(*batch))

        state_batch = torch.tensor(batch[0], device=device, dtype=torch.float)
        reward_batch = torch.tensor(batch[1], device=device)
        next_state_batch = torch.tensor(batch[2], device=device, dtype=torch.float)
        done_batch = torch.tensor(batch[3], device=device)

        return state_batch, reward_batch, next_state_batch, done_batch

