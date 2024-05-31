# import torch
import pygame
import sys
from enum import Enum
from collections import namedtuple
import random

from snake import Snake, Direction, BLOCK_SIZE, Color, Point
from agent import Agent, Agents

WIDTH = 640
HEIGHT = 480
FPS = 1000


class Game:
    def __init__(self,
                 n_agents=500,
                 genetic=True):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple testing game")

        self.clock = pygame.time.Clock()
        font = pygame.font.Font("arial.ttf", 25)
        self.n_agents = n_agents
        self.genetic = genetic

        if genetic:
            self.agents = Agents(w=WIDTH,
                                 h=HEIGHT,
                                 n_agents=n_agents,
                                 screen=self.screen,
                                 font=font)

        else:
            self.snake = Snake(w=WIDTH, h=HEIGHT)
            self.snake.display = self.screen
            self.snake.font = font

            self.agent = Agent(snake=self.snake, n_games=self.n_games)

    def display_blocks(self):
        for x in range(0, WIDTH, BLOCK_SIZE):
            for y in range(0, HEIGHT, BLOCK_SIZE):
                pygame.draw.rect(self.screen,
                                 Color.PURPLE.value,
                                 pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))

                # This is the width
                z = 0.2
                pygame.draw.rect(self.screen,
                                 Color.BLACK.value,
                                 pygame.Rect(x+z, y+z, BLOCK_SIZE - 2*z, BLOCK_SIZE - 2*z))

    def play(self):
        if self.genetic:
            self.run_genetic()
        else:
            self.run()

    def run_genetic(self,):

        # run for each game
        for game_no in range(self.agents.total_games):
            games_played = 0
            print("####################")
            print(f"GAME NO: {game_no}")

            if game_no > 0:
                self.agents.next_generation()
            print(f"Total agents are {len(self.agents.agents)}")

            # call for the next gen
            while True:

                if (len(self.agents.agents)) == games_played:
                    break

                # now run each snake
                self.agent = self.agents.get_current_agent()
                self.current_agent = self.agent["agent"]

                # simulate 1 snake game
                while True:

                    # if the snake is stuck in a loop
                    if self.agents.curr_steps > 100 and self.current_agent.snake.score < 10:
                        self.current_agent.snake.game_over = True

                    if self.current_agent.snake.game_over:
                        # print(f"agent: {self.agent['agent_no']}")
                        self.agents.update_agent()

                        # reset agent's snake for next fresh run
                        self.current_agent.snake.reset()

                        # update no of games played
                        games_played += 1
                        break

                    self.screen.fill(Color.BLACK.value)
                    self.display_blocks()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()

                    self.current_agent.get_state()
                    self.current_agent.take_action()
                    self.current_agent.snake.update()
                    self.current_agent.snake.render()

                    self.agents.update_steps()

                    pygame.display.flip()
                    self.clock.tick(FPS)
        print("\nResults\n#############\n")
        for i in range(len(self.agents.agents.keys())):
            a = self.agents.agents[i]
            print(f"Agent: {a['agent_no']}")
            print(f"Fitness: {a['fitness']}")
            print(f"Steps {a['steps']}")
            print(f"Score {a['score']}")
            print()

    def run(self, ai=False):

        if not ai:
            while not self.snake.game_over:
                self.screen.fill(Color.BLACK.value)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            self.snake.head_direction = Direction.LEFT
                        elif event.key == pygame.K_RIGHT:
                            self.snake.head_direction = Direction.RIGHT
                        elif event.key == pygame.K_UP:
                            self.snake.head_direction = Direction.UP
                        elif event.key == pygame.K_DOWN:
                            self.snake.head_direction = Direction.DOWN

                self.display_blocks()
                self.snake.update()
                self.snake.render()
                pygame.display.flip()

                # self.agent.get_state()
                # self.agent.take_action()

                self.clock.tick(FPS)

            print(f"Final Score: {self.snake.score}")
        else:
            while self.n_games:
                if self.snake.game_over:
                    # print(f'The fitness of the snake is {self.agent.fitness()}')
                    # print(f"The socre of the snake is {self.snake.score}")
                    self.snake.reset()
                self.screen.fill(Color.BLACK.value)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                self.display_blocks()

                self.agent.get_state()
                self.agent.take_action()

                self.snake.update()
                self.snake.render()
                pygame.display.flip()
                self.clock.tick(FPS)

            print(f"Final Score: {self.snake.score}")


Game().play()
