# import torch
import pygame
import sys
from enum import Enum
from collections import namedtuple
import random

from snake import Snake, Direction, BLOCK_SIZE, Color, Point
from agent import Agent

WIDTH = 640
HEIGHT = 480
FPS = 20


class Game:
    def __init__(self,
                 n_games=100):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple testing game")

        self.clock = pygame.time.Clock()
        self.snake = Snake(w=WIDTH, h=HEIGHT)
        self.snake.display = self.screen
        font = pygame.font.Font("arial.ttf", 25)
        self.snake.font = font
        self.n_games = n_games

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

Game().run(ai=True)
