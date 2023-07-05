import pygame


class TowerDefenseGame:
    def __init__(self):
        pygame.init()

        self.WIDTH, self.HEIGHT = 800, 800
        self.GRID_SIZE = 10
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.FPS = 60

        self.WHITE = (255, 255, 255)
        self.GREY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

    def draw_grid(self):
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                rect = pygame.Rect(j * self.CELL_SIZE, i * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[i][j] == "tower":
                    pygame.draw.rect(self.WIN, self.RED, rect)
                elif self.grid[i][j] == "wall":
                    pygame.draw.rect(self.WIN, self.BLUE, rect)
                else:
                    pygame.draw.rect(self.WIN, self.GREY, rect, 1)

    def main(self):
        clock = pygame.time.Clock()
        run = True

        while run:
            clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    i, j = y // self.CELL_SIZE, x // self.CELL_SIZE
                    if event.button == 1:
                        if all(self.grid[i + di][j + dj] == "" for di in range(-1, 1) for dj in range(-1, 1)):
                            for di in range(-1, 1):
                                for dj in range(-1, 1):
                                    self.grid[i + di][j + dj] = "tower"
                    elif event.button == 3:
                        if self.grid[i][j] == "":
                            self.grid[i][j] = "wall"

            self.draw_grid()
            pygame.display.update()

        pygame.quit()
