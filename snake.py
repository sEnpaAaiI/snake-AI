import pygame
from collections import namedtuple
from enum import Enum
import random


pygame.init()
font = pygame.font.Font("arial.ttf", 25)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


BLOCK_SIZE = 20
CONSTANT_SPEED = 1

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


class Snake:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2, BLUE)
        self.snake = [self.head,
                      Point(self.w/2 - BLOCK_SIZE,
                            self.h/2 , BLUE),
                      Point(self.w/2 - 2 * BLOCK_SIZE, self.h/2, BLUE)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y, RED)
        if self.food in self.snake:
            self._place_food()

    def __update_ui(self, surf):
        for p in self.snake:
            pygame.draw.rect(surf,
                             p.color,
                             pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            y = 4
            pygame.draw.rect(surf,
                             PURPLE,
                             pygame.Rect(p.x+y, p.y+y, BLOCK_SIZE - 2*y, BLOCK_SIZE- 2*y))
            del y

        text = font.render("Score: " + str(self.score), True, WHITE)
        surf.blit(text, [0, 0])
        pygame.draw.rect(surf, 
                         self.food.color,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

    def update(self, surf):
        self.__update_ui(surf)
