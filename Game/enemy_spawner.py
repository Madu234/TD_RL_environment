import pygame
from enemy import Enemy

normal_characteristics = {"width":0.5,"height":0.5,"color":(0, 255, 0),"speed":3,"hp":50,"armor":0}
armored_characteristics = {"width":0.75,"height":0.75,"color":(0, 255, 255),"speed":2, "hp":50,"armor":5}

class EnemySpawner:
    def __init__(self, path, enemy_type, start_point, end_point, enemy_number, enemy_frequency, cell_size, game_FPS):
        self.enemy_type = enemy_type 
        self.start_point = start_point
        self.end_point = end_point
        self.enemy_number = enemy_number
        self.enemy_frequency = enemy_frequency
        self.cell_size = cell_size
        self.path = path
        self.spawned_enemies = 0
        # self.last_spawn_time = pygame.time.get_ticks()
        self.last_spawn_frame = 0


    def spawn(self, current_frame):
        # current_time = pygame.time.get_ticks()
        # if self.spawned_enemies < self.enemy_number and current_time - self.last_spawn_time >= self.enemy_frequency:
        if self.spawned_enemies < self.enemy_number and current_frame - self.last_spawn_frame >= self.enemy_frequency:
            #self.last_spawn_time = current_time
            self.last_spawn_frame = current_frame
            self.spawned_enemies += 1
            if self.enemy_type == 1:
                return Enemy(path = self.path, 
                             width = self.cell_size * normal_characteristics['width'], 
                             height = self.cell_size * normal_characteristics['height'], 
                             color = normal_characteristics['color'], 
                             speed = normal_characteristics['speed'], 
                             cell_size = self.cell_size, 
                             hp = normal_characteristics['hp'],
                             armor = normal_characteristics['armor'])
            elif self.enemy_type == 2:
                return Enemy(path = self.path, 
                             width = self.cell_size * armored_characteristics['width'], 
                             height = self.cell_size * armored_characteristics['height'], 
                             color = armored_characteristics['color'], 
                             speed = armored_characteristics['speed'], 
                             cell_size = self.cell_size, 
                             hp = armored_characteristics['hp'],
                             armor = armored_characteristics['armor'])
            
            return Enemy(path = self.path, width = self.cell_size // 2, height = self.cell_size // 2, color=(0, 255, 0), speed=3, cell_size=self.cell_size, hp = 50)
        else:
            return None

    def get_spawns_left(self):
        return self.enemy_number - self.spawned_enemies