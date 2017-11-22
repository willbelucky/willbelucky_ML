# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 18.
"""
import copy
import random
import sys

Maze = [[0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0]]

gamma = 0.7
l_rate = 0.0001
player = []
objection = []
trap = [0, 4]
tragectory = []
MoveDirection = []
Q_table = []
reward = 0


def initialize(q_table):
    for xp in range(5):
        q_table.append([])
        for yp in range(5):
            q_table[xp].append([])
            for direction in range(4):
                q_table[xp][yp].append(1)


def print_info():
    s = ""

    for x in range(5):
        for y in range(5):
            if [x, y] == player:
                s = s + "P "
                continue
            elif [x, y] == objection:
                s = s + "O "
                continue
            elif [0, 4] == [x, y]:
                s = s + "X "
                continue
            else:
                if Maze[x][y] == 0:
                    s = s + "- "
                else:
                    s = s + "+ "
        s = s + "\n"

    return s


def move(q_table, player):
    total = 0

    m_prob = []
    for x in range(4):
        value = q_table[player[0]][player[1]][x]
        total = total + value

    move_amount = 0.0
    for x in range(4):
        value = q_table[player[0]][player[1]][x]
        move_amount = move_amount + float(value) / float(total)
        m_prob.append(move_amount)

    move = random.random()
    next_position = -1

    for x in range(4):
        if move < m_prob[x]:
            next_position = x
            break

    return next_position


def check_move(move):
    if move == 0:  # Move <<
        if player[0] == 0:
            return -1
        else:
            tmp = player[0] - 1
            if Maze[tmp][player[1]] == 1:
                return -1
            player[0] = tmp
            return 1

    elif move == 1:  # Move >>
        if player[0] == 4:
            return -1
        else:
            tmp = player[0] + 1
            if Maze[tmp][player[1]] == 1:
                return -1
            player[0] = tmp
            return 1

    elif move == 2:  # Move up
        if player[1] == 0:
            return -1
        else:
            tmp = player[1] - 1
            if Maze[player[0]][tmp] == 1:
                return -1
            player[1] = tmp
            return 1

    elif move == 3:  # Move down
        if player[1] == 4:
            return -1
        else:
            tmp = player[1] + 1
            if Maze[player[0]][tmp] == 1:
                return -1
            player[1] = tmp
            return 1
    else:
        print("Error!!!")
        sys.exit()


def check_goal(player):
    if player == objection:
        return 100, True

    if player == trap:
        return -100, True

    else:
        return -1, False


def learning(tragectory, move_direction, reward, gamma):
    chk_count = 0
    while 1:

        [xp, yp] = tragectory[-1]
        direction = move_direction[-1]

        [nx, ny] = [0, 0]
        if direction == 0:
            [nx, ny] = [xp - 1, yp]
        elif direction == 1:
            [nx, ny] = [xp + 1, yp]
        elif direction == 2:
            [nx, ny] = [xp, yp - 1]
        elif direction == 3:
            [nx, ny] = [xp, yp + 1]
        else:
            print("Error")

        dot_max = -1
        for x in range(4):
            if Q_table[nx][ny][x] > dot_max:
                dot_max = Q_table[nx][ny][x]

        Q_table[xp][yp][direction] += l_rate * (gamma * dot_max - Q_table[xp][yp][direction])

        if chk_count == 0:
            Q_table[xp][yp][direction] += l_rate * (reward ** 3)
            chk_count += 1

        if len(move_direction) < 2:
            break

        tragectory.pop()
        move_direction.pop()


if __name__ == '__main__':
    initialize(Q_table)
    outer_iteration = 0

    while 1:

        player = [0, 0]
        objection = [4, 4]
        trap = [0, 4]

        total_reward = 0
        add = copy.deepcopy(player)
        tragectory.append(add)
        inner_iteration = 0
        while 1:
            next_position = move(Q_table, player)
            chk = check_move(next_position)
            if chk == -1:
                continue

            add = copy.deepcopy(player)
            tragectory.append(add)
            MoveDirection.append(next_position)

            # if outer_iteration % 5000 == 1 and outer_iteration > 100:
            #     print(print_info())
            #     input()

            reward, do_break = check_goal(player)
            total_reward += reward
            inner_iteration += 1
            if do_break:
                # print(total_reward)
                break

        if outer_iteration % 5000 == 1 and outer_iteration > 100:
            print("current score : {}, reward : {}".format(inner_iteration, total_reward))

        tragectory.pop()  # We does't need last objective goal

        learning(tragectory, MoveDirection, reward, gamma)
        tragectory = []
        MoveDirection = []
        outer_iteration += 1

        if outer_iteration % 1000 == 0:
            print("Iteration %d done." % outer_iteration)
