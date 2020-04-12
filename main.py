import environment as env
from agent import Agent, load_agent
import torch
import numpy as np
import math
import pickle
import sys

num_iter = 50000000
print_interval = 10
save_interval = 200

agent = Agent(num_actions=5) if len(sys.argv) == 1 else load_agent(sys.argv[1])
#agent.optimizer = torch.optim.Adam(agent.local_Q.parameters(), 5e-4)

print(agent.optimizer)

for episode in range(agent.start, num_iter):
    done = False
    score = 0
    ep_duration = 0
    board, state = env.reset()
    while not done:
        next_states = env.process_state(board)
        action = agent.select_action(next_states)
        board, reward, done = env.step(board, action)
        agent.store_experience(state, reward, next_states[action], 1-done)
        state = next_states[action]
        score += reward
        ep_duration += 1

    agent.learn()
    agent.episodes.append(episode)
    agent.scores.append(score)
    agent.durations.append(ep_duration)
    agent.start = episode
    
    if episode % print_interval == 0 or ep_duration > 800:
        avg_score = np.mean(agent.scores[max(0, episode-print_interval):(episode+1)])
        avg_duration = np.mean(agent.durations[max(0, episode-print_interval):(episode+1)])
        if episode % save_interval == 0:
            agent.start = episode + 1
            agent.save(str(episode))
        print("Episode: %d - Avg. Duration: %d - Avg. Score: %.3f - Epsilon %.3f" % 
                    (episode, avg_duration, avg_score, agent.eps_start))
