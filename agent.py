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
                 output_units=4):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ELU(),
            # nn.Linear(hidden_units * 2, hidden_units),
            # nn.ELU(),
            nn.Linear(hidden_units, output_units),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.l(state)

# class SnakeModel:
#     def __init__(self,
#                  input_units, 
#                  hidden_units,
#                  output_units):
#         self.l1 = np.random.randn(input_units, hidden_units)
#         self.l2 = np.random.randn(hidden_units, output_units)

#     def forward(self, state):
#         # 32 X 1, 32 X 10 
#         x = state.T @ self.l1 # 1, 10
#         return x @ self.l2 # 1, 10 ... 10, 4 -> 1, 4

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
                 snake,
                 n_games=10,
                 model=SnakeModel):
        self.n_games = n_games
        self.state = None
        self.snake = snake
        self.model = model(32, 10, 4)

    def get_state(self) -> None:
        """
        one hot vectors of head and tail direction
        8 directions which include 3 things
        - distance from the wall (variable)
        - distance of the food (variable)
        - is snake body in between (variable)
        """
        # gather head direction
        head_direction = self.snake.head_direction
        tail_direction = self.snake.tail_direction

        directions = [Direction.RIGHT, Direction.LEFT,
                      Direction.UP, Direction.DOWN]

        # converting to one-hot
        head_direction = [1 if head_direction == x else 0 for x in directions]
        tail_direction = [1 if tail_direction == x else 0 for x in directions]

        # horizontal direction
        # we have to give a fn which defines how to get next block to self.snake__direction_state, and it will return
        # all the three values.
        def rd(b):
            """right direction"""
            x, y = b.x, b.y
            return Point(x+BLOCK_SIZE, y, Color.BLACK.value)

        def ld(b):
            """left direction"""
            x, y = b.x, b.y
            return Point(x-BLOCK_SIZE, y, Color.BLACK.value)

        def ud(b):
            """up direction"""
            x, y = b.x, b.y
            return Point(x, y-BLOCK_SIZE, Color.BLACK.value)

        def dd(b):
            """down direction"""
            x, y = b.x, b.y
            return Point(x, y+BLOCK_SIZE, Color.BLACK.value)

        def t1(b):
            x, y = b.x, b.y
            return Point(x+BLOCK_SIZE, y+BLOCK_SIZE, Color.BLACK.value)

        def t2(b):
            x, y = b.x, b.y
            return Point(x+BLOCK_SIZE, y-BLOCK_SIZE, Color.BLACK.value)

        def t3(b):
            x, y = b.x, b.y
            return Point(x-BLOCK_SIZE, y+BLOCK_SIZE, Color.BLACK.value)

        def t4(b):
            x, y = b.x, b.y
            return Point(x-BLOCK_SIZE, y-BLOCK_SIZE, Color.BLACK.value)

        # horizontal direction (right)
        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, rd)
        right_direction = [wall_distance, snake_body, food_distance]

        # horizontal direction (left)
        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, ld)
        left_direction = [wall_distance, snake_body, food_distance]

        # vertical direction (up)
        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, ud)
        up_direction = [wall_distance, snake_body, food_distance]

        # vertical direction (down)
        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, dd)
        down_direction = [wall_distance, snake_body, food_distance]

        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, t1)
        t1_direction = [wall_distance, snake_body, food_distance]

        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, t2)
        t2_direction = [wall_distance, snake_body, food_distance]

        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, t3)
        t3_direction = [wall_distance, snake_body, food_distance]

        wall_distance, snake_body, food_distance = self.snake._direction_state(
            self.snake.head, t4)
        t4_direction = [wall_distance, snake_body, food_distance]
        # making tensor
        self.state = torch.tensor([
            *right_direction,
            *left_direction,
            *up_direction,
            *down_direction,
            *t1_direction,
            *t2_direction,
            *t3_direction,
            *t4_direction,
            *head_direction,
            *tail_direction
        ],
            dtype=torch.float32,)

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
        return next_direction

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
