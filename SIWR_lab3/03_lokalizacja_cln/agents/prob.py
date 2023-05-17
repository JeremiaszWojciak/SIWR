# prob.py
# This is

import random
import numpy as np

from gridutil import *

best_turn = {('N', 'E'): 'turnright',
             ('N', 'S'): 'turnright',
             ('N', 'W'): 'turnleft',
             ('E', 'S'): 'turnright',
             ('E', 'W'): 'turnright',
             ('E', 'N'): 'turnleft',
             ('S', 'W'): 'turnright',
             ('S', 'N'): 'turnright',
             ('S', 'E'): 'turnleft',
             ('W', 'N'): 'turnright',
             ('W', 'E'): 'turnright',
             ('W', 'S'): 'turnleft'}


class LocAgent:

    def __init__(self, size, walls, dir, eps_perc, eps_move):
        self.size = size
        self.walls = walls
        # list of valid locations
        self.locations = list({*locations(self.size)}.difference(self.walls))
        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.dir = dir
        self.eps_perc = eps_perc
        self.eps_move = eps_move
        # whether to plan next direction to move
        self.plan_next_move = True
        # planned direction
        self.next_dir = None

        # previous action
        self.prev_action = None

        self.t = 0

        # uniform posterior over valid locations
        prob = 1.0 / len(self.locations)
        self.P = prob * np.ones([len(self.locations)], dtype=np.float)
        # print('P: ', self.P)
        # print('P shape: ', self.P.shape)

    def __call__(self, percept):
        # update posterior
        # TODO PUT YOUR CODE HERE
        if self.prev_action == 'forward':
            T = np.zeros((42, 42))
            for loc in self.locations:
                new_loc, _ = self.forward(loc, self.dir)
                if new_loc in self.locations:
                    T[self.loc_to_idx[loc], self.loc_to_idx[loc]] = 0.05
                    T[self.loc_to_idx[new_loc], self.loc_to_idx[loc]] = 0.95
                else:
                    T[self.loc_to_idx[loc], self.loc_to_idx[loc]] = 1.0
        else:
            T = np.eye(42, 42, dtype=float)

        O = np.zeros((42))
        for loc in self.locations:
            p = 1.0
            for dir in ['N', 'E', 'S', 'W']:
                neighbor, _ = self.forward(loc, dir)
                if neighbor in self.locations and dir not in percept:
                    p *= 0.9
                elif neighbor not in self.locations and dir in percept:
                    p *= 0.9
                else:
                    p *= 0.1
            O[self.loc_to_idx[loc]] = p

        self.P = (1 / np.sum(T @ self.P * O)) * (T @ self.P * O)

        # ------------------

        if self.plan_next_move:
            # randomly choose from not obstacles
            dirs = [d for d in ['N', 'E', 'S', 'W'] if d not in percept]
            if len(dirs) == 0:
                dirs.append('N')
            self.next_dir = random.choice(dirs)
            self.plan_next_move = False

        action = 'forward'
        if self.dir != self.next_dir:
            action = best_turn[(self.dir, self.next_dir)]
        else:
            self.plan_next_move = True

        if action == 'turnleft':
            self.dir = leftTurn(self.dir)
        elif action == 'turnright':
            self.dir = rightTurn(self.dir)

        self.prev_action = action

        return action

    def getPosterior(self):
        P_arr = np.zeros([self.size, self.size], dtype=np.float)
        for idx, loc in enumerate(self.locations):
            P_arr[loc[0], loc[1]] = self.P[idx]
        return P_arr

    def forward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    def backward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    @staticmethod
    def turnright(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 1) % 4
        return cur_loc, dirs[idx]

    @staticmethod
    def turnleft(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 4 - 1) % 4
        return cur_loc, dirs[idx]
