import pygame
from enemy import Enemy
class EnemySpawner:
    def __init__(self, path, enemy_type, start_point, end_point, enemy_number, enemy_frequency, cell_size):
        self.enemy_type = enemy_type
        self.start_point = start_point
        self.end_point = end_point
        self.enemy_number = enemy_number
        self.enemy_frequency = enemy_frequency
        self.cell_size = cell_size
        self.path = path
        self.spawned_enemies = 0
        self.last_spawn_time = pygame.time.get_ticks()

    def spawn(self):
        current_time = pygame.time.get_ticks()
        if self.spawned_enemies < self.enemy_number and current_time - self.last_spawn_time >= self.enemy_frequency:
            self.last_spawn_time = current_time
            self.spawned_enemies += 1
            return Enemy(path = self.path, width = self.cell_size // 2, height = self.cell_size // 2, color=(0, 255, 0), speed=3, cell_size=self.cell_size, hp = 50)
        else:
            return None
