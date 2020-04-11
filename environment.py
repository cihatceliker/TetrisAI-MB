import os
import numpy as np
import random
import sys
from dataclasses import dataclass

ROW = 20
COL = 10

EMPTY = -1
PIECE = -2
GROUND = -3

# ARS rotation
SHAPES = {
    0: [
        [(-1,-1),(-1,0),(0,-1),(0,0)]   # O
    ],
    2: [
        [(0,-2),(0,-1),(0,0),(0,1)],    # I
        [(-1,-1),(-1,0),(0,0),(0,1)],   # Z
        [(0,-1),(-1,0),(0,0),(-1,1)]    # Z'
    ],
    4: [
        [(-1,-1),(0,-1),(0,0),(0,1)],   # L'
        [(0,-1),(0,0),(0,1),(-1,1)],    # L
        [(0,-1),(-1,0),(0,0),(0,1)]     # T
    ]
}
ROT_L = [
    [0,1],
    [-1,0]
]
# n E [0-3]: rotated n times
ALL_SHAPES = {
    0: [],
    1: [],
    2: [],
    3: []
}

def rotate_n_times(shape, n):
    new_shape = []
    for coor in shape:
        new_coor = coor
        for k in range(n):
            new_coor = np.dot(new_coor, ROT_L)
        new_shape.append(tuple(new_coor))
    return new_shape

for n in range(4):
    ALL_SHAPES[n].append(SHAPES[0][0])
    for shape in SHAPES[2]:
        ALL_SHAPES[n].append(rotate_n_times(shape, n%2))
    for shape in SHAPES[4]:
        ALL_SHAPES[n].append(rotate_n_times(shape, n))

##### game methods
def reset():
    return add_new_piece(Board(
        area=np.ones((ROW, COL)) * EMPTY
    ))

def drop(old_board, _):
    board = old_board.clone()
    while board.valid:
        board = move(board, (1,0))
    return board

def clear_complete_lines(old_board):
    board = old_board.clone()
    idxs = []
    for i in range(ROW-1, 0, -1):
        if not EMPTY in board.area[i,:]:
            idxs.append(i)
    complete_lines = len(idxs)
    for idx in reversed(idxs):
        board.area[1:idx+1,:] = board.area[0:idx,:]
    return board, complete_lines

def rotate(old_board, clockwise):
    board = old_board.clone()
    to = 1 if clockwise else -1
    board.rotation_idx = (board.rotation_idx + to) % 4
    if is_available(board):
        board.rotation_idx = old_board.rotation_idx
        board = make(board, EMPTY)
        board.rotation_idx = (board.rotation_idx + to) % 4
        board = make(board, PIECE)
    else:
        board.rotation_idx = old_board.rotation_idx
    return board

def is_available(board, to=(0,0)):
    for i, j in ALL_SHAPES[board.rotation_idx][board.piece_idx]:
        x, y = i+to[0]+board.rel_x, j+to[1]+board.rel_y
        if x >= ROW or y < 0 or y >= COL:
            return False
        if board.area[x, y] == GROUND:
            return False
    return True

def make(old_board, num):
    board = old_board.clone()
    for i, j in ALL_SHAPES[board.rotation_idx][board.piece_idx]:
        board.area[board.rel_x+i,board.rel_y+j] = num
    return board

def add_new_piece(old_board, drop_point=(1,5)):
    board = old_board.clone()
    board.rel_x, board.rel_y = drop_point
    board.piece_idx = np.random.randint(7)
    board.rotation_idx = 0
    if not is_available(board):
        board.done = True
    else:
        board = make(board, PIECE)
    return board

def move(old_board, to):
    board = old_board.clone()
    valid = is_available(board, to)
    if valid:
        board = make(board, EMPTY)
        board.rel_x += to[0]
        board.rel_y += to[1]
        board = make(board, PIECE)
    else:
        board.valid = False
    return board

actions = {
    0: (lambda x, y: x, None),
    1: (move, (0,-1)),
    2: (move, (0,1)),
    3: (rotate, False),
    4: (rotate, True),
    5: (drop, None)
}

def step(old_board, action):
    board = old_board.clone()
    board = actions[action][0](board, actions[action][1])

    reward = 0
    board.valid = True
    board = move(board, (1,0))

    if not board.valid:
        board = make(board, GROUND)
        board, complete_lines = clear_complete_lines(board)
        aggregate_height, bumpiness, holes = analyze(board)
        print("AGG_HEIGHT %d, BUMPY %d, HOLES %d" % (aggregate_height, bumpiness, holes))
        board = add_new_piece(board)

    return board, reward, board.done, ""

def analyze(board):
    aggregate_height = 0
    holes = 0
    bumpiness = 0
    heights = np.zeros(COL)
    for j in range(COL):
        for i in range(ROW):
            if board.area[i,j] == GROUND:
                heights[j] = ROW - i
                break
    aggregate_height = sum(heights)
    for j in range(COL):
        piece_found = False
        if j < COL-1:
            bumpiness += abs(heights[j]-heights[j+1])
        for i in range(ROW):
            if board.area[i,j] == GROUND:
                piece_found = True
            if piece_found and board.area[i,j] == EMPTY:
                holes += 1
    return aggregate_height, bumpiness, holes

@dataclass
class Board:
    area: np.ndarray
    piece_idx: int = 0
    rotation_idx: int = 0
    rel_x: int = 0
    rel_y: int = 0
    done: bool = False
    valid: bool = True
    def clone(self):
        return Board(
            area=self.area.copy(),
            piece_idx=self.piece_idx,
            rotation_idx=self.rotation_idx,
            rel_x=self.rel_x,
            rel_y=self.rel_y,
            done=self.done,
            valid=True
        )
