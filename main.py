import pygame
import sys
from enum import Enum
from collections import namedtuple
import random

from snake import Snake, Direction, BLOCK_SIZE, Color

WIDTH = 640
HEIGHT = 480
FPS = 15

class Game:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple testing game")

        self.clock = pygame.time.Clock()
        self.snake = Snake(w=WIDTH, h=HEIGHT)
        self.snake.display = self.screen
        font = pygame.font.Font("arial.ttf", 25)
        self.snake.font = font

    def run(self):

        while not self.snake.game_over:
            self.screen.fill(Color.BLACK.value)
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
                        self.snake.direction = Direction.DOWN

            self.snake.update()
            self.snake.render()
            # pygame.display.update()
            pygame.display.flip()
            self.clock.tick(FPS)

        print(f"Final Score: {self.snake.score}")


Game().run()
