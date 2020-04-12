import time
import threading
import numpy as np
import random
import pickle
import environment as env
from agent import load_agent
from tkinter import Frame, Canvas, Tk
import sys
import pickle

COLORS = {
    env.EMPTY: "#fff",
    env.PIECE: "#5900ff",
    env.GROUND: "#3e2175"
}

class GameGrid():

    def __init__(self, speed=0.02, size=720):
        width = size / 2
        height = size
        self.root = Tk()
        self.root.configure(background=COLORS[env.EMPTY])
        self.game = Canvas(self.root, width=width, height=height, bg=COLORS[env.EMPTY])
        self.game.pack()
        self.speed = speed
        self.size = size
        self.rectangle_size = size/env.ROW
        self.pause = False
        self.image_counter = 0
        self.init()
        self.root.title('Tetris')

        self.commands = {
            113: 1, # Left
            114: 2, # Right
            53: 3, # Z
            52: 3, # X
            65: 5, # Drop
            37: 0 # Do nothing
        }
        self.root.bind("<Key>", self.key_down)
        self.agent = load_agent(sys.argv[1])

        threading.Thread(target=self.watch_play).start()
        #threading.Thread(target=self.play).start()
        self.root.mainloop()

    def watch_play(self):
        self.agent.eps_start = 0
        while True:
            done = False
            ep_duration = 0
            board, state = env.reset()
            while not done:
                next_actions, next_states = zip(*env.process_state(board).items())
                action = self.agent.select_action(next_states)
                next_state = next_states[action]
                board, reward, done = env.step(board, next_actions[action])
                ep_duration += 1
                self.board = board.area
                self.update()
                
                time.sleep(self.speed)

    def update(self):
        for i in range(env.ROW):
            for j in range(env.COL):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = COLORS[curr]
                if i == 3:
                    color="#aaa"
                self.game.itemconfig(rect, fill=color)

    def init(self):
        def draw(x1, y1, sz, color, func):
            return func(x1, y1, x1+sz, y1+sz, fill=color, width=0)
        self.game_area = []
        for i in range(env.ROW):
            row = []
            for j in range(env.COL):
                color = COLORS[env.EMPTY]
                rect = draw(j*self.rectangle_size, i*self.rectangle_size, 
                            self.rectangle_size, color, self.game.create_rectangle)
                row.append(rect)
            self.game_area.append(row)


    def key_down(self, event):
        self.pause = False
    
    def play(self):
        self.action = 0
        while True:
            done = False
            board, state = env.reset()
            while not done:
                if not self.pause:
                    self.pause = True
                    states = env.process_state(board)
                    next_state = states[action]
                    print(state)
                    print(next_state)
                    
                    board, reward, done = env.step(board, action)
                    self.action = 0
                    self.board = board.area
                    self.update()

if __name__ == "__main__":
    GameGrid()
