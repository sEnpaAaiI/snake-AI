import torch
from snake import Direction, Point, Color, BLOCK_SIZE
from model import SnakeModel


class Agent:
    def __init__(self,
                 snake,
                 model=SnakeModel):
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
        variable -> distance measured in blocks.
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
        # we have to give a fn which defines how to get next block to
        # self.snake__direction_state, and it will return
        # all the required three values.

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
        # right_direction = [wall_distance, snake_body, food_distance]
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
        directions = [Direction.RIGHT,
                      Direction.LEFT, Direction.UP, Direction.DOWN]
        next_direction = directions[next_action]
        self.snake.head_direction = next_direction

    def fitness(self, steps):
        """f(steps, apples) = steps + (2**apples + apples**2.1*500) - [apples ** 1.2 * (0.25 * steps)**1.3]"""

        apples = self.snake.score
        return 1000 * apples + 1 * steps
        return steps + (2 ** apples + apples ** 2.1 * 500) - (apples ** 1.2 * (0.25 * steps) ** 1.3)
