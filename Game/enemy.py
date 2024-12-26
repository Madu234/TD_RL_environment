import pygame
# from Game import TowerDefenseGame

class Enemy:
    def __init__(self, path, width, height, color, speed, cell_size, hp, armor = 0):
        
        # Type variables
        self.width = width
        self.height = height
        self.color = color
        self.speed = speed
        # self.type = enemy_type
        self.armor = armor

        # Path variables

        self.path = path
        self.current_target = 0
        self.cell_y, self.cell_x = self.path[self.current_target]
        self.end_point = path[-1]
        
        self.cell_size = cell_size

        # Compute the pixel coordinates
        self.x = self.cell_x * cell_size + cell_size // 2
        self.y = self.cell_y * cell_size + cell_size // 2

        self.hp = hp  # Initialize HP

    def is_alive(self):
        return self.hp > 0

    def take_damage(self, damage):
        if (damage > self.armor):
            self.hp = self.hp - (damage - self.armor)
        else:
            # Take 0 damage
            pass

    def shred_armor(self, shred):
        self.armor = self.armor - shred

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x - self.width // 2, self.y - self.height // 2, self.width, self.height))

    def move(self):
        target_x, target_y = self.path[self.current_target + 1]

        if self.cell_x < target_x:
            self.cell_x += self.speed / self.cell_size
            if self.cell_x > target_x:  # Fix overshooting
                self.cell_x = target_x
        elif self.cell_x > target_x:
            self.cell_x -= self.speed / self.cell_size
            if self.cell_x < target_x:  # Fix overshooting
                self.cell_x = target_x

        if self.cell_y < target_y:
            self.cell_y += self.speed / self.cell_size
            if self.cell_y > target_y:  # Fix overshooting
                self.cell_y = target_y
        elif self.cell_y > target_y:
            self.cell_y -= self.speed / self.cell_size
            if self.cell_y < target_y:  # Fix overshooting
                self.cell_y = target_y

        # Check if reached target cell
        if self.cell_x == target_x and self.cell_y == target_y:
            # print (f"{self.cell_x} and {self.cell_y}")
            self.current_target += 1
            if self.current_target == len(self.path) - 1:
                self.hp=0
                # print("DAMAGE")
                # game.player_life = game.player_life - 1
                return 1 # Enemy has reached the final destination, you may take any required action

        # Recompute the pixel coordinates
        self.x = self.cell_x * self.cell_size + self.cell_size // 2
        self.y = self.cell_y * self.cell_size + self.cell_size // 2

