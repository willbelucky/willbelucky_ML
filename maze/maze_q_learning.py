# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 18.
"""
import random

gamma = 0.7  # discount factor

MAZE = [[0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0]]

LENGTH = len(MAZE)

Network = []
Player = 0
Target = LENGTH * LENGTH - 1
Trap = LENGTH - 1
q_table = []
Reward = []
l_rate = 1.0
gamma = 0.7
r_target = 100.
r_trap = -100.


def make_network(length):
    for x in range(length * length):
        Network.append([])

    for x in range(length):
        for y in range(length):
            num = x * length + y

            if y != 0 and MAZE[x][y - 1] == 0:
                Network[num].append(num - 1)

            if y % length != length - 1 and MAZE[x][y + 1] == 0:
                Network[num].append(num + 1)

            if x != 0 and MAZE[x - 1][y] == 0:
                Network[num].append(num - length)

            if x != length - 1 and MAZE[x + 1][y] == 0:
                Network[num].append(num + length)


def print_info():
    s = ""

    px = Player % LENGTH
    py = Player / LENGTH

    ox = LENGTH - 1
    oy = LENGTH - 1

    for x in range(LENGTH):
        for y in range(LENGTH):
            if [x, y] == [py, px]:
                s = s + "P "
                continue
            elif [x, y] == [oy, ox]:
                s = s + "O "
                continue
            elif [x, y] == [0, LENGTH - 1]:
                s = s + "X "
                continue
            else:
                if MAZE[x][y] == 0:
                    s = s + "- "
                else:
                    s = s + "+ "
        s = s + "\n"

    return s


def prob():
    s = ""

    px = Player % LENGTH
    py = Player / LENGTH

    ox = 4
    oy = 4

    for x in range(LENGTH):
        for y in range(LENGTH):

            max_v = max(q_table[x * LENGTH + y])
            for i in range(len(q_table[x * LENGTH + y])):
                if max_v == q_table[x * LENGTH + y][i]:
                    max_add = i

            if Network[x * LENGTH + y][max_add] == x * LENGTH + y - 1:
                direction = "L"
            elif Network[x * LENGTH + y][max_add] == x * LENGTH + y + 1:
                direction = "R"
            elif Network[x * LENGTH + y][max_add] == x * LENGTH + y - LENGTH:
                direction = "U"
            elif Network[x * LENGTH + y][max_add] == x * LENGTH + y + LENGTH:
                direction = "D"
            else:
                raise ValueError("Network[x * L + y][max_add] {} is not matched.\n".format(Network[x * LENGTH + y][max_add])
                                 + "x = {}, L = {}, y = {}".format(x, LENGTH, y))

            if [x, y] == [py, px]:
                s = s + "%.1f(%s)\t" % (max_v, direction)
                continue
            elif [x, y] == [oy, ox]:
                s = s + "%.1f(%s)\t" % (max_v, direction)
                continue
            elif [x, y] == [0, 4]:
                s = s + "%.1f(%s)\t" % (max_v, direction)
                continue
            else:
                if MAZE[x][y] == 0:
                    s = s + "%.1f(%s)\t" % (max_v, direction)
                else:
                    s = s + "+\t"
        s = s + "\n"

    return s


def initialize(q_table):
    for state in range(LENGTH * LENGTH):
        q_table.append([])
        for direction in range(len(Network[state])):
            q_table[state].append(0.)
            if state == Target:
                q_table[state][direction] = r_target
            if state == Trap:
                q_table[state][direction] = r_trap


def find_next(q_table, network, player, action):
    candidate_state = network[player][action]
    max_q = max(q_table[candidate_state])

    max_add = q_table[candidate_state].index(max_q)
    candidate_action = max_add

    if max_q == 0.:
        max_add = random.randrange(0, len(network[candidate_state]))
        candidate_action = max_add

    return candidate_action


if __name__ == '__main__':
    make_network(LENGTH)
    initialize(q_table)

    for itr in range(101):
        Player = 0
        action = random.randrange(0, len(Network[Player]))
        print(itr)
        while 1:

            if itr == 100:
                print(print_info())
                print(prob())
                input()
            if Player == Target or Player == Trap:
                break
            next_state = Network[Player][action]
            next_action = find_next(q_table, Network, Player, action)
            q_table[Player][action] += l_rate * (gamma * q_table[next_state][next_action] - q_table[Player][action])

            Player = next_state
            action = next_action
