import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import pickle

device = "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
print("runs on %s." % device)
device = torch.device(device)


class Agent():
    
    def __init__(self, eps_start=1.0, eps_decay=0.996, gamma=0.99, 
                memory_capacity=20000, batch_size=512, alpha=7e-3):
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.eps_start = eps_start
        self.eps_decay = eps_decay
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
                action = torch.argmax(self.network(obs)).item()
        else:
            action = np.random.randint(len(states))
        return action

    def learn(self):
        if self.batch_size >= len(self.replay_memory.memory):
            return
        
        state_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_memory.sample(self.batch_size)
        
        prediction = self.network(state_batch)[:,0]
        with torch.no_grad():
            evaluated = self.network(next_state_batch)[:,0]
            evaluated = reward_batch + self.gamma * evaluated * done_batch

        self.optimizer.zero_grad()
        self.loss(prediction, evaluated).to(device).backward()
        self.optimizer.step()

        self.eps_start = max(0, self.eps_decay * self.eps_start)
        
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

