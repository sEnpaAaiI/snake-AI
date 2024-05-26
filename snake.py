import pygame
from collections import namedtuple
from enum import Enum
import random


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Color(Enum):
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

BLOCK_SIZE = 20
CONSTANT_SPEED = 1

Point = namedtuple("Point", ['x', 'y', 'color'])


class Snake:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2, Color.BLUE.value)
        self.snake = [self.head,
                      Point(self.w/2 - BLOCK_SIZE,
                            self.h/2, Color.BLUE.value),
                      Point(self.w/2 - 2 * BLOCK_SIZE, self.h/2, Color.BLUE.value)]

        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y, Color.RED.value)
        if self.food in self.snake:
            self._place_food()

    def __move(self, direction):
        x = self.head.x
        y = self.head.y

        # right, left, up, down
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y, Color.BLUE.value)

    def __update_ui(self):
        for p in self.snake:
            pygame.draw.rect(self.display,
                             p.color,
                             pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            y = 4
            pygame.draw.rect(self.display,
                             Color.PURPLE.value,
                             pygame.Rect(p.x+y, p.y+y, BLOCK_SIZE - 2*y, BLOCK_SIZE - 2*y))
            # del y

        pygame.draw.rect(self.display,
                         self.food.color,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = self.font.render("Score: " + str(self.score), True, Color.WHITE.value)
        self.display.blit(text, [0, 0])

    def check_game_over(self):
        # check if the snake is outside the window
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            self.game_over = True
            return

        # check if collision with itself.
        if self.head in self.snake[1:]:
            self.game_over = True
            return

    def update(self):
        # update the snake head to include the next head
        self.__move(self.direction)

        # update the snake to include the head, insert at start
        self.snake.insert(0, self.head)

        # check for game over
        self.check_game_over()
        if self.game_over:
            return
        # if food is eaten then keep the snake as it is if not then pop element
        if self.head.x == self.food.x and self.head.y == self.food.y:
            self._place_food()
            self.score += 1
        else:
            self.snake.pop()

    def render(self):
        self.__update_ui()
