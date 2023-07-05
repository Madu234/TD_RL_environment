import pygame

class Enemy:
    def __init__(self, cell_x, cell_y, width, height, color, speed, cell_size):
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.width = width
        self.height = height
        self.color = color
        self.speed = speed
        self.cell_size = cell_size

        # Compute the pixel coordinates
        self.x = self.cell_x * cell_size + cell_size // 2
        self.y = self.cell_y * cell_size + cell_size // 2

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x - self.width // 2, self.y - self.height // 2, self.width, self.height))

    def move(self):
        # Increase the cell y-coordinate
        self.cell_y += self.speed / self.cell_size

        # Recompute the pixel coordinates
        self.y = self.cell_y * self.cell_size + self.cell_size // 2