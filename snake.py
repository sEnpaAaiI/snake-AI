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

Point = namedtuple("Point", ['x', 'y', 'color'])


class Snake:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        self.reset()

    def reset(self):
        self.head_direction = Direction.RIGHT
        self.tail_direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2, Color.BLUE.value)
        self.snake = [self.head,
                      Point(self.w/2 - BLOCK_SIZE,
                            self.h/2, Color.BLUE.value),
                      Point(self.w/2 - 2 * BLOCK_SIZE, self.h/2, Color.BLUE.value)]

        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False
        self.tail = self.snake[-1]

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

    def _direction_state(self, intial_box, get_next_block):
        """
        inital_box: Point()
        The distance given is in terms of blocks
        """
        wall_distance, snake_body, food_distance = 0, 0, 0
        # this is to keep track how far in terms of blocks we are from the initial_box
        curr_blocks = 0
        next_block = get_next_block(intial_box)
        while True:
            if (next_block.x >= self.w) or (next_block.y >= self.h) or (next_block.x <= 0) or (next_block.y <= 0):
                break
            wall_distance += 1
            curr_blocks += 1

            # not considering head cause we are looking from head, i.e., initial box is head itself
            # slight optimization, as there is only 1 food currently.
            if food_distance == 0:
                if next_block.x == self.food.x and next_block.y == self.food.y:
                    food_distance = curr_blocks

            if snake_body == 0:
                for body in self.snake[1:]:
                    if body.x == next_block.x and body.y == next_block.y:
                        snake_body = curr_blocks

            next_block = get_next_block(next_block)
        return (wall_distance, snake_body, food_distance)

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
        text = self.font.render(
            "Score: " + str(self.score), True, Color.WHITE.value)
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
        self.__move(self.head_direction)

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

        # update tail_direction
        self.tail_direction = self.check_relative_block_position(
            self.tail, self.snake[-1])
        self.tail = self.snake[-1]

    def check_relative_block_position(self, block1, block2):
        """
        It will return relative pos of block2 wrt block1
        eg: if block2 is in left to block1 it will return left..
        """
        if block1.x < block2.x:
            return Direction.RIGHT
        if block1.x > block2.x:
            return Direction.LEFT
        if block1.y < block2.y:
            return Direction.DOWN
        if block1.y > block2.y:
            return Direction.UP

    def render(self):
        self.__update_ui()
