import pygame
import sys
from snake import Snake

WIDTH = 640
HEIGHT = 480
FPS = 15

# COlORS
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Game:
    def __init__(self):
        pygame.init()

        pygame.display.set_caption("Simple testing game")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.snake = Snake(w=WIDTH,
                           h=HEIGHT)

    def run(self):

        while True:
            self.screen.fill(BLACK)

            # right, left, up, down
            movement = [0, 0, 0, 0]

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN or event.key == pygame.K_w:
                        movement[2] = 1
                    if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        movement[3] = 1
                    if event.key == pygame.K_DOWN or event.key == pygame.K_a:
                        movement[1] = 1
                    if event.key == pygame.K_DOWN or event.key == pygame.K_d:
                        movement[0] = 1

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        movement[2] = 0
                    if event.key == pygame.K_UP or event.key == pygame.K_s:
                        movement[3] = 0
                    if event.key == pygame.K_UP or event.key == pygame.K_a:
                        movement[1] = 0
                    if event.key == pygame.K_UP or event.key == pygame.K_d:
                        movement[0] = 0

            # self.snake.update(movement)
            self.snake.update(self.screen)
            # self.snake.render(self.screen)
            pygame.display.update()
            self.clock.tick(FPS)


Game().run()
