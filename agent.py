import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Optional, Dict
from snake import Direction, Point, Color, BLOCK_SIZE, Snake


class Agents:
    def __init__(self,
                 w,
                 h,
                 screen,
                 font,
                 n_agents=10,):
        self.w = w
        self.h = h
        self.agents = dict()
        self.screen = screen
        self.font = font
        self.gen = 0
        for i in range(n_agents):
            self.agents[i] = {
                "agent": Agent(snake=Snake(w=w, h=h),
                               model=SnakeModel),
                "fitness": list(),
                "color": Color.GREEN.value,
                "steps": 0,
                "gen": self.gen,
                "score": 0,
                "agent_no": i,
            }

        # calculate the games played
        temp = len(self.agents)
        self.total_games = 0
        while temp != 0:
            self.total_games += 1
            top = temp // 2
            if top == 1:
                self.total_games += 1
                del top
                break
            if top % 2 != 0:
                top -= 1
            temp = top
        del temp

        print(f"Total games to play: {self.total_games}")
        self.__initialize()

    def __initialize(self):
        # setting some values
        for i in range(len(self.agents)):
            self.agents[i]["agent"].snake.display = self.screen
            self.agents[i]["agent"].snake.font = self.font

        self.current_agent_idx = 0
        self.curr_steps = 0

    def get_current_agent(self):
        self.current_agent_idx += 1
        return self.agents[self.current_agent_idx-1]

    def update_agent(self):
        temp = self.agents[self.current_agent_idx-1]
        temp["steps"] = self.curr_steps
        temp["fitness"] = temp["agent"].fitness(self.curr_steps)
        temp["score"] = temp["agent"].snake.score
        self.curr_steps = 0

    def update_steps(self):
        self.curr_steps += 1

    def sort_agents(self) -> dict:
        """
        Sorts agents in descending order based on their fitness
        """
        return dict(
            sorted(
                self.agents.items(),
                key=lambda item: -item[1]["fitness"]
            )
        )

    def __combine_agents(self, agents):
        new_agents = dict()
        idx = 0
        for a in range(0, len(agents), 2):
            try:
                # create new agent
                m = Agent(snake=Snake(w=self.w,
                                      h=self.h),
                                      model=SnakeModel)

                # print("made agent?????????")
                # get each agent's model
                m1 = agents[a]["agent"].model
                m2 = agents[a+1]["agent"].model
                new_m = m.model

                # update the new agents weights
                for (n1, a1), (n2, a2), (n3, a3) in zip(m1.named_parameters(), m2.named_parameters(), new_m.named_parameters()):
                    with torch.no_grad():
                        a3.copy_(a1 * 0.5 + a2 * 0.5)

                new_agents[idx] = {
                    "agent": m,
                    "fitness": list(),
                    "color": Color.GREEN.value,
                    "steps": list(),
                    "gen": self.gen,
                    "score": list(),
                    "agent_no": idx,
                }
                idx += 1
            except Exception as e:
                # this is when there aren't two value or
                # when len(agents) is odd
                print(f"Some excpetion occured for: {idx}")
                # print(e)
                pass

        return new_agents

    def next_generation(self):
        """
        Take current agents (n) and mutate them into n/2 or n/2-1

        Steps:
        - sort the agents based on fitness.
        - combine weights of 2 agents
        - make new agent from the combined weights
        - update the agents list
        """
        self.gen += 1

        sorted_agents = self.sort_agents()
        combined_agents = self.__combine_agents(agents=sorted_agents)
        self.agents = combined_agents
        self.__initialize()

    def some(self):
        pass


class SnakeModel(nn.Module):
    """
    the output is interpreted as follows
    right, left, up, down
    """

    def __init__(self,
                 input_units,
                 hidden_units,
                 output_units = 4):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, output_units),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.l(state)
        return x

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


class Agent:
    def __init__(self,
                 snake,
                 model = SnakeModel):
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
            dtype=torch.float32,).reshape(1, -1)

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
        self.snake.head_direction = next_direction
        # print(f"snake will move in {self.snake.head_direction}")

    def fitness(self, steps):
        """f(steps, apples) = steps + (2**apples + apples**2.1*500) - [apples ** 1.2 * (0.25 * steps)**1.3]"""
        apples = self.snake.score

        return steps + (2 ** apples + apples ** 2.1 * 500) - (apples ** 1.2 * (0.25 * steps) ** 1.3)
