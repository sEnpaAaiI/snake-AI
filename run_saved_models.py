import pygame
import os
import sys
from collections import namedtuple
import numpy as np

from snake import BLOCK_SIZE, Color, Point
from agent import Agents

WIDTH = 400
HEIGHT = 400
FPS = 30


class Game:
    def __init__(self,
                 n_agents=1000,
                 total_games=1000):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI genetic Algorithm")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("arial.ttf", 25)
        self.n_agents = n_agents
        self.total_games = total_games

    def display_blocks(self):
        """
        Displays blocks in the game screen
        """
        for x in range(0, WIDTH, BLOCK_SIZE):
            for y in range(0, HEIGHT, BLOCK_SIZE):
                pygame.draw.rect(self.screen,
                                 Color.PURPLE.value,
                                 pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))

                # This is the width for inner rectangle.
                z = 0.2
                pygame.draw.rect(self.screen,
                                 Color.BLACK.value,
                                 pygame.Rect(x+z, y+z, BLOCK_SIZE - 2*z, BLOCK_SIZE - 2*z))

    def extra_display(self, gen):
        """
        Displays extra information
        """
        text = self.font.render(
            "GEN: " + str(gen), True, Color.WHITE.value)
        self.screen.blit(text, [WIDTH - 7 * BLOCK_SIZE, 0])

    def save_best_agent(self, game_no):
        """
        Saves the model weigths (i.e., np array)
        """
        save_dir = f'weights/{game_no}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # best score model
        l1 = self.agents.best_scores["best_score_agent"].model.l1
        l1_b = self.agents.best_scores["best_score_agent"].model.l1_b

        l2 = self.agents.best_scores["best_score_agent"].model.l2
        l2_b = self.agents.best_scores["best_score_agent"].model.l2_b

        np.save(f"weights/{game_no}/_score_l1", l1)
        np.save(f"weights/{game_no}/_score_l1_b", l1_b)

        np.save(f"weights/{game_no}/_score_l2", l2)
        np.save(f"weights/{game_no}/_score_l2_b", l2_b)

        # best fitness model
        l1 = self.agents.best_scores["best_fitness_agent"].model.l1
        l1_b = self.agents.best_scores["best_fitness_agent"].model.l1_b

        l2 = self.agents.best_scores["best_fitness_agent"].model.l2
        l2_b = self.agents.best_scores["best_fitness_agent"].model.l2_b

        np.save(f"weights/{game_no}/_fitness_l1", l1)
        np.save(f"weights/{game_no}/_fitness_l1_b", l1_b)

        np.save(f"weights/{game_no}/_fitness_l2", l2)
        np.save(f"weights/{game_no}/_fitness_l2_b", l2_b)

        print("Model weights saved...")

    def load_weights(self, game_no, model):
        model.l1 = np.load(f"weights/{game_no}/_score_l1.npy")
        model.l1_b = np.load(f"weights/{game_no}/_score_l1_b.npy")

        model.l2 = np.load(f"weights/{game_no}/_score_l2.npy")
        model.l2_b = np.load(f"weights/{game_no}/_score_l2_b.npy")

        return 

    def play(self, flag=0):
        if flag == 0:

            flag = 1
            from _agent import Agent
            from snake import Snake

            self.snake = Snake(w=WIDTH, h=HEIGHT)
            self.snake.display = self.screen
            self.snake.font = self.font

            self.agent = Agent(snake=self.snake)
            self.load_weights(138, self.agent.model)

        while True:
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


Game().play()
