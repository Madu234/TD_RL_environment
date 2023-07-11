# enemy_spawner.py
import pygame
from enemy import Enemy

class EnemySpawner:
    def __init__(self, enemy_type, enemy_number, enemy_frequency, cell_size):
        self.enemy_type = enemy_type
        self.enemy_number = enemy_number
        self.enemy_frequency = enemy_frequency
        self.cell_size = cell_size

        self.spawned_enemies = 0
        self.last_spawn_time = pygame.time.get_ticks()

    def spawn(self):
        current_time = pygame.time.get_ticks()
        if self.spawned_enemies < self.enemy_number and current_time - self.last_spawn_time >= self.enemy_frequency:
            self.last_spawn_time = current_time
            self.spawned_enemies += 1
            return Enemy(0, 0, self.cell_size // 2, self.cell_size // 2, color=(0, 255, 0), speed=3, cell_size=self.cell_size)
        else:
            return None
