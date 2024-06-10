import pygame
import sys
from collections import namedtuple

from snake import BLOCK_SIZE, Color, Point
from agent import Agents

WIDTH = 240
HEIGHT = 240
FPS = 1000


class Game:
    def __init__(self,
                 n_agents=10,
                 total_games=100):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI genetic Algorithm")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("arial.ttf", 25)
        self.n_agents = n_agents
        self.total_games = total_games
        self.agents = Agents(w=WIDTH,
                             h=HEIGHT,
                             n_agents=n_agents,
                             screen=self.screen,
                             font=self.font)

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

    def play(self,):
        """
        Defines logic to run the genetic algorithm
        """

        def __print_scores():
            self.agents.next_generation()
            print("####################")
            print(f"GAME NO: {game_no}")
            print(f"Best Scores:")
            for k, v in self.agents.best_scores.items():
                print(f"{k}: {v}")
            
            print(f"This GEN Best Scores:")
            for k, v in self.agents.this_gen_best_scores.items():
                print(f"{k}: {v}")
            print("\n")

        # run for each game
        for game_no in range(self.total_games):
            games_played_by_curr_gen = -1
            if game_no != 0:
                __print_scores()
                # print(f"Total agents are {len(self.agents.agents)}")

            else: 
                print(f"Starting Simulation...")

            # Run the simulation for on Generation
            while True:

                games_played_by_curr_gen += 1
                if games_played_by_curr_gen == self.n_agents:
                    break
                
                # Get snake to run simulation on
                self.agent = self.agents.get_current_agent()
                self.current_agent = self.agent["agent"]

                # simulate the snake
                while True:

                    # if the snake is stuck in a loop
                    if self.agents.curr_steps > 100 and self.current_agent.snake.score < 10:
                        self.current_agent.snake.game_over = True

                    if self.current_agent.snake.game_over:
                        self.agents.update_agent()

                        # reset agent's snake for next fresh run
                        self.current_agent.snake.reset()
                        break

                    self.screen.fill(Color.BLACK.value)
                    self.display_blocks()
                    self.extra_display(self.agents.gen)

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

        # print("\nResults\n#############\n")
        # for i in range(len(self.agents.agents.keys())):
        #     a = self.agents.agents[i]
        #     print(f"Agent: {a['agent_no']}")
        #     print(f"Fitness: {a['fitness']}")
        #     print(f"Steps {a['steps']}")
        #     print(f"Score {a['score']}")
        #     print()
        __print_scores()

    def temp(self, flag=0):
        if flag == 0:

            flag = 1
            from _agent import Agent
            from snake import Snake

            self.snake = Snake(w=WIDTH, h=HEIGHT)
            self.snake.display = self.screen
            self.snake.font = self.font

            self.agent = Agent(snake=self.snake)

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