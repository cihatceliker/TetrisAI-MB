import time
import threading
import numpy as np
import random
import pickle
import environment as env
from tkinter import Frame, Canvas, Tk
import sys
from pyscreenshot import grab
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

        #threading.Thread(target=self.watch_play).start()
        threading.Thread(target=self.play).start()
        self.root.mainloop()

    def watch_play(self):
        while True:
            duration = 0
            done = False
            board = env.reset()
            while not done:
                action = np.random.randint(6)
                board, reward, done, next_piece = env.step(board, action)
                self.board = board.area
                self.update()
                duration += 1
                time.sleep(self.speed)

    def update(self, rel_x=0, rel_y=0):
        for i in range(env.ROW):
            for j in range(env.COL):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = COLORS[curr]
                if rel_x == i and rel_y == j:
                    color = "#000"
                self.game.itemconfig(rect, fill=color)

    def watch_history(self):
        for state in self.processed:
            self.board = state
            self.update()
            time.sleep(self.speed)

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
        if event.keycode == 24: # q
            self.quit = True
        if event.keycode in self.commands:
            self.action = self.commands[event.keycode]
            self.pause = False
    
    def play(self):
        self.action = 0
        while True:
            done = False
            board, states = env.reset()
            while not done:
                if not self.pause:
                    self.pause = True
                    env.process_state(board)
                    board, reward, done = env.step(board, self.action)
                    self.action = 0
                    self.board = board.area
                    self.update(board.rel_x, board.rel_y)

if __name__ == "__main__":
    GameGrid()
