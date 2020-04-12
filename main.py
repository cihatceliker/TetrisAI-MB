import environment as env
from agent import Agent, load_agent
import torch
import numpy as np
import math
import pickle
import sys

num_iter = 50000000
print_interval = 10
save_interval = 100

agent = Agent() if len(sys.argv) == 1 else load_agent(sys.argv[1])
#agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 5e-4)
print(agent.optimizer)

def count(tetrises):
    s = {}
    for i in range(1,5):
        s[i] = len([j for j in tetrises if j == i])
    return str(s)

all_tetrises = []
for episode in range(agent.start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    board, state = env.reset()
    while not done:
        next_actions, next_states = zip(*env.process_state(board).items())
        action = agent.select_action(next_states)
        next_state = next_states[action]
        board, reward, done = env.step(board, next_actions[action])
        agent.store_experience(state, reward, next_state, 1-done)
        state = next_state
        score += reward
        ep_duration += 1
    agent.learn()
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode
    all_tetrises += board.tetrises
    
    if episode % print_interval == 0 or ep_duration > 800:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.start = episode + 1
            agent.save(str(episode))
        
        print("Episode: %d - Avg. Duration: %d - Avg. Score: %3.3f - %s" % (episode, avg_duration, avg_score, count(all_tetrises)))
        all_tetrises = []
        
