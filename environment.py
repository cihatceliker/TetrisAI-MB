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

def reset():
    board = Board(
        area=np.ones((ROW, COL)) * EMPTY,
        piece_idx=np.random.randint(7)
    )
    board.tetrises = []
    return board, (0,)*4

def drop(old_board):
    board = old_board.clone()
    board = make(board, EMPTY)
    while is_available(board, (1,0)):
        board.rel_x += 1
    board = make(board, GROUND)
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

def add_drop_analyze(old_board, drop_point, rotation_idx, piece_idx):
    board = old_board.clone()
    board.rel_x, board.rel_y = drop_point
    board.piece_idx = piece_idx
    board.rotation_idx = rotation_idx
    board = drop(board)
    board, complete_lines = clear_complete_lines(board)
    aggregate_height, bumpiness, holes = analyze(board)
    return board, complete_lines, aggregate_height, bumpiness, holes

def step(old_board, action):
    board = old_board.clone()
    drop_col, rot_idx = action
    board, complete_lines, aggregate_height, bumpiness, holes \
            = add_drop_analyze(board, (1, drop_col), rot_idx, board.piece_idx)
    
    reward = 1+(complete_lines ** 2) * 10
    
    if complete_lines != 0:
        board.tetrises.append(complete_lines)        

    if GROUND in board.area[2]:
        reward -= 2
        board.done = True

    board.piece_idx = np.random.randint(7)
    return board, reward, board.done

def process_state(old_board):
    before_drop = analyze(old_board)
    states = {}
    board = old_board.clone()
    for i in range(4):
        for j in range(COL):
            board.rotation_idx = i
            board.rel_x = 1
            board.rel_y = j
            if is_available(board):
                states[(j,i)] = drop_analyze(board, (1,j), i, *before_drop)
        if i == 0 and old_board.piece_idx == 0:
            break
        if i == 1 and old_board.piece_idx < 4:
            break
    return states

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
        board = Board(
            area=self.area.copy(),
            piece_idx=self.piece_idx,
            rotation_idx=self.rotation_idx,
            rel_x=self.rel_x,
            rel_y=self.rel_y,
            done=self.done,
            valid=True
        )
        board.tetrises = self.tetrises
        return board

def drop_analyze(old_board, drop_point, rotation_idx, before_agg, before_bum, before_holes):
    board, complete_lines, aggregate_height, bumpiness, holes \
        = add_drop_analyze(old_board, drop_point, rotation_idx, old_board.piece_idx)
    return complete_lines, \
           holes-before_holes, \
           bumpiness-before_bum, \
           aggregate_height-before_agg