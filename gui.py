import time
import environment as env
from tkinter import Frame, Canvas, Tk

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
        self.init()
        self.watch_play()

    def watch_play(self):
        done = False
        board = env.reset()
        while not done:
            # finds the best action
            action = env.process_state(board)
            self.drop_piece(action, board)
            board, done = env.step(board, *action)

            self.board = board.area
            self.update()
            self.root.update()
            time.sleep(self.speed)

    # this method drops piece a move at a time. its just to see the 
    # piece slide down instead of directly dropping it. just for viewing purposes
    def drop_piece(self, action, old_board):
        board = old_board.clone()
        board.rel_x, board.rel_y, board.rotation_idx = 1, *action
        while env.is_available(board, (1,0)):
            board = env.make(board, env.EMPTY)
            board.rel_x += 1
            board = env.make(board, env.PIECE)
            self.board = board.area
            self.update()
            self.root.update()
            time.sleep(self.speed)

    # update colors
    def update(self):
        for i in range(env.ROW):
            for j in range(env.COL):
                rect = self.game_area[i][j]
                curr = int(self.board[i, j])
                color = COLORS[curr]
                self.game.itemconfig(rect, fill=color)

    # init colors
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
