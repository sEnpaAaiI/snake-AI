import pygame
import sys
from enum import Enum
from collections import namedtuple
import random

from snake import Snake, Direction, BLOCK_SIZE

WIDTH = 640
HEIGHT = 480
FPS = 15

# COlORS
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)
LIGHT_GRAY = (211, 211, 211)
DARK_GRAY = (169, 169, 169)
BROWN = (165, 42, 42)
PINK = (255, 192, 203)

Point = namedtuple("Point", ['x', 'y', 'color'])


class Game:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple testing game")

        self.clock = pygame.time.Clock()
        self.snake = Snake(w=WIDTH, h=HEIGHT)
        # self.snake = SnakeGame()
        self.snake.display = self.screen
        font = pygame.font.Font("arial.ttf", 25)
        self.snake.font = font

    def run(self):

        while True:
            self.screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.snake.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.snake.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        self.snake.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        print("pressed")
                        self.snake.direction = Direction.DOWN

            self.snake.update()
            self.snake.render()
            # pygame.display.update()
            pygame.display.flip()
            self.clock.tick(FPS)


Game().run()
