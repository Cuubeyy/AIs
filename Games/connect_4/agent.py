import time
from collections import deque
import random

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchviz import make_dot

from Game import Game
from model import QTrainer, Linear_QNet

MAX_MEMORY = 100_000
BATCH_SIZE = 1024


class Agent:
    def __init__(self):
        self.games = 0
        self.epsilon = 0.2
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(42 * 3, 256, 7)
        self.trainer = QTrainer(self.model, 0.01, self.gamma)

    def get_state(self, game):
        state = game.get_state()
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: torch.Tensor):
        move = [0 for _ in range(7)]
        if random.random() < self.epsilon:
            slot = random.randint(0, 6)
        else:
            prediction = self.model(state)
            slot = torch.argmax(prediction).item()
        move[slot] = 1

        return torch.tensor(move, dtype=torch.float)

    def get_agent_move(self, game):
        state_old = self.get_state(game)
        move = self.get_action(state_old)
        converted_move = torch.argmax(move)
        while not game.is_move_possible(converted_move):
            move = self.get_action(state_old)
            converted_move = torch.argmax(move)
        return converted_move, move


def train():
    agent = Agent()
    game = Game()
    winner_list = []
    iterations = 0
    while True:
        state_old = agent.get_state(game)
        converted_move, move = agent.get_agent_move(game)
        reward, done = game.game_step(converted_move, False)

        state_new = agent.get_state(game)
        # agent.train_short_memory(state_old, move, reward, state_new, done)
        agent.remember(state_old, move, reward, state_new, done)

        if done:
            iterations += 1
            # train long memory, plot result
            agent.games += 1
            if iterations % 10 == 0:
                agent.train_long_memory()
                # print("Iteration:", iterations)
                eps = agent.epsilon
                agent.epsilon = 0.05
                winner_list.append(play_game(agent))
                agent.epsilon = eps
                print(sum(winner_list) / len(winner_list))
                if len(winner_list) > 20:
                    winner_list = winner_list[-20:]
            game.reset()


def play_game(agent: Agent):
    winner = []
    game = Game()

    for round in range(10):
        for player_one in [True, False]:
            game.reset()
            while True:
                player_one = not player_one
                if player_one:
                    while not game.is_move_possible(random.randint(0, 6)):
                        pass
                else:
                    converted_move, _ = agent.get_agent_move(game)
                    while not game.is_move_possible(converted_move):
                        pass
                if game.game_step(game.last_moves[-1], False)[1]:
                    # print("Winner is:", "agent" if not player_one else "random")
                    winner.append(not player_one)
                    break
    return sum(winner) / len(winner)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
