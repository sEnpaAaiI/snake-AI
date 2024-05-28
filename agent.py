import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Optional, Dict
from snake import Direction, Point, Color, BLOCK_SIZE
# import pygame


class SnakeModel(nn.Module):
    """
    the output is interpreted as follows
    right, left, up, down
    """

    def __init__(self,
                 input_units,
                 hidden_units,
                 output_units):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(input_units, hidden_units * 2),
            nn.ELU(),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units * 2, output_units)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.l(state)


class SnakeTrainer:
    def __init__(self,
                 lr=0.001):
        self.model = SnakeModel()
        self.lr = lr

    def train_per_iteration(self,
                            curr_state,
                            next_state):
        pass


class Agent:
    def __init__(self,
                 n_games,
                 snake,
                 model=SnakeModel):
        self.n_games = n_games
        self.state = None
        self.snake = snake
        self.model = model()

    def get_state(self, snake):
        self.state = ...
        pass

    def take_action(self):
        """
        Updates the next direction 
        of snake.
        """
        next_action = self.model(self.state)
        next_action = torch.argmax(next_action).item()
        temp_direction = [Direction.RIGHT,
                          Direction.LEFT, Direction.UP, Direction.DOWN]
        next_direction = temp_direction[next_action]
        self.snake.direction = next_direction

    def loss_fn(self):
        """
        Implements the loss function or the 
        quality function
        """
        pass

    def single_train_step(self,):
        """
        Trains the ai for steps
        """
        pass

    def train_batch(self):
        """
        Train the ai for batches
        """
        pass