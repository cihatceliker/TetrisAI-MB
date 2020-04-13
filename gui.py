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

    def __init__(self, speed=0.01, size=720):
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
        self.history = load_agent(sys.argv[1])
        
        #self.agent = load_agent(sys.argv[1])
        #threading.Thread(target=self.watch_play).start()
        threading.Thread(target=self.watch_history).start()
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

                ep_duration += self.drop_piece(next_actions[action], board)

                next_state = next_states[action]
                board, reward, done = env.step(board, next_actions[action])
                
                self.board = board.area
                self.update()                
                time.sleep(self.speed)
            print(ep_duration)

    def drop_piece(self, actions, old_board):
        board = old_board.clone()
        board.rel_x, board.rel_y, board.rotation_idx = 1, *actions
        count = 0
        while env.is_available(board, (1,0)):
            board = env.make(board, env.EMPTY)
            board.rel_x += 1
            board = env.make(board, env.PIECE)
            self.board = board.area
            self.update()
            time.sleep(self.speed)
            count += 1
        return count

    def watch_history(self):
        for state in self.history:
            self.board = state
            self.update()
            time.sleep(self.speed)

    def update(self):
        for i in range(env.ROW):
            for j in range(env.COL):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = COLORS[curr]
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


if __name__ == "__main__":
    GameGrid()
