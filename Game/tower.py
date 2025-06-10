import pygame
import math

normal_tower = {'range':100,'Reload_time':20,'damage':10,'max_age':0,'dmg_mod_per_age':0,'rld_mod_per_age':0,'color':(255, 0, 0)}
age_tower = {'range':100,'Reload_time':20,'damage':10,'max_age':5,'dmg_mod_per_age':5,'rld_mod_per_age':2,'color':(200, 255, 0)}

class Projectile:
    def __init__(self, start_pos, target, damage, armor_shred = 0):
        self.x, self.y = start_pos
        self.target = target
        self.damage = damage
        self.armor_shred = armor_shred
        self.speed = 5  # You can adjust this value
        self.color = (0, 0, 0)  # Projectile color - currently black
        self.radius = 5  # Radius of the projectile circle
        self.hit_reward = 0.1

    def move(self):
        direction = (self.target.x - self.x, self.target.y - self.y)
        distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        direction = (direction[0] / distance, direction[1] / distance)
        self.x += direction[0] * self.speed
        self.y += direction[1] * self.speed
        if math.hypot(self.target.x - self.x, self.target.y - self.y) < self.radius:
            self.target.shred_armor(self.armor_shred)
            self.target.take_damage(self.damage)     
            return True  # Indicates that the projectile hit the target

        return False  # Projectile has not hit the target yet

    def draw(self, WIN):
        pygame.draw.circle(WIN, self.color, (int(self.x), int(self.y)), self.radius)

class Tower:
    def __init__(self, pos, cell_size, range, Reload_time, max_age = 0, dmg_mod_per_age = 0, rld_mod_per_age = 0,type = 0):
        self.i = pos[0]
        self.j = pos[1]
        self.x = (pos[0]+1) * cell_size
        self.y = (pos[1]+1) * cell_size
        self.target = None
        #Aging
        self.range = range
        self.Reload_time = 0
        
        self.projectiles = []
        self.color = normal_tower['color']
        self.damage = normal_tower['damage'] 
        self.aging_left = max_age
        self.damage_modification_per_age = normal_tower['dmg_mod_per_age']  
        self.reloaded_modification_per_age = normal_tower['rld_mod_per_age']
        self.aging_left = normal_tower['max_age']

        if type==0:
            self.range = normal_tower['range']
            self.Reload_time = int(normal_tower['Reload_time'])
            
            self.projectiles = []
            self.color = normal_tower['color']
            self.damage = normal_tower['damage'] 
            self.aging_left = max_age
            self.damage_modification_per_age = normal_tower['dmg_mod_per_age']  
            self.reloaded_modification_per_age = normal_tower['rld_mod_per_age']
            self.aging_left = normal_tower['max_age']
        elif type == 1:
            self.range = age_tower['range']
            self.Reload_time = int(age_tower['Reload_time'])
            self.projectiles = []
            self.color = age_tower['color']
            self.damage = age_tower['damage']
            self.damage_modification_per_age = age_tower['dmg_mod_per_age']  
            self.reloaded_modification_per_age = age_tower['rld_mod_per_age']
            self.aging_left = age_tower['max_age']

        self.width = 20  # Width of the tower
        self.height = 20  # Height of the tower
        self.shots_fired = 0
        self.last_shot_frame = 0
    def aging(self):
        #print(self.aging_left)
        if self.aging_left > 0:
            self.damage += self.damage_modification_per_age
            #print(self.damage)
            self.Reload_time -+ self.reloaded_modification_per_age
        self.aging_left -= 1

    def center_position(self):
        return (self.x + self.width / 2, self.y + self.height / 2)

    def find_target(self, enemies):
        if not self.target or not self.target.is_alive() or not self.is_in_range(self.target):
            closest_enemy = None
            closest_dist = self.range + 1
            for enemy in enemies:
                dist = ((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2) ** 0.5
                if dist < closest_dist:
                    closest_enemy = enemy
                    closest_dist = dist

            self.target = closest_enemy
    
    def reset_shoot_count(self):
        self.shots_fired = 0

    def is_in_range(self, enemy):
        return ((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2) ** 0.5 <= self.range

    def shoot(self, current_frame):
        if self.target and current_frame - self.last_shot_frame >= self.Reload_time:
            self.projectiles.append(Projectile((self.x, self.y), self.target, self.damage))
            self.last_shot_frame = current_frame
            self.shots_fired += 1

    def get_color(self):
        return self.color

    def update_projectiles(self):
        total_hit_reward = 0
        new_projectiles = []
        for projectile in self.projectiles:
            if not projectile.move():
                new_projectiles.append(projectile)
            else:
                total_hit_reward += projectile.hit_reward
        self.projectiles = new_projectiles
        return total_hit_reward