import pygame

WIDTH = 640
HEIGHT = 480


class Game:
    def __init__(self):
        pygame.init()

        pygame.display.set_caption("Simple testing game")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.rectangle = pygame.Rect(50, 50, 300, 50)
        self.rec_pos = [160, 260]

    def run(self):
        while True:
            self.screen.fill((12, 219, 248))
            self.rec_pos = [x + y for x, y in zip(self.rec_pos, [1, 1])]
            self.n_r = pygame.Rect(10, 10, *self.rec_pos)
            pygame.draw.rect(self.screen, (0, 100, 100), 
                             self.n_r)

            pygame.display.update()
            self.clock.tick(60)


Game().run()
