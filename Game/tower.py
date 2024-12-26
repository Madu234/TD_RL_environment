import pygame
import math

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
        # Calculate direction towards the target
        direction = (self.target.x - self.x, self.target.y - self.y)
        distance = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        direction = (direction[0] / distance, direction[1] / distance)

        # Move projectile
        self.x += direction[0] * self.speed
        self.y += direction[1] * self.speed

        # Check if the projectile has hit the target
        if math.hypot(self.target.x - self.x, self.target.y - self.y) < self.radius:
            self.target.shred_armor(self.armor_shred)
            self.target.take_damage(self.damage)  
            # self.hit_reward += self.reward_per_hit         
            return True  # Indicates that the projectile hit the target

        return False  # Projectile has not hit the target yet

    def draw(self, WIN):
        pygame.draw.circle(WIN, self.color, (int(self.x), int(self.y)), self.radius)

class Tower:
    def __init__(self, pos, cell_size, range, attack_speed, game_FPS):
        self.x = (pos[0]+1) * cell_size
        self.y = (pos[1]+1) * cell_size

        self.range = range
        self.attack_speed = int(attack_speed * game_FPS / 60)
        self.target = None
        self.projectiles = []
        self.color = (255, 0, 0)  # Color of the tower
        self.width = 20  # Width of the tower
        self.height = 20  # Height of the tower
        self.damage = 10

        # For targeting priotization
        # self.armor_threshold
        self.shots_fired = 0
        self.last_shot_frame = 0
        #self.last_shot_time = pygame.time.get_ticks()

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
        test = 0
        # current_time = pygame.time.get_ticks()
        # if self.target and current_time - self.last_shot_time > 1000 / self.attack_speed:
        if self.target and current_frame - self.last_shot_frame >= self.attack_speed:
            self.projectiles.append(Projectile((self.x, self.y), self.target, self.damage))
            self.last_shot_frame = current_frame
            #self.last_shot_time = current_time
            self.shots_fired += 1

    def update_projectiles(self):
        total_hit_reward = 0
        new_projectiles = []
        for projectile in self.projectiles:
            if not projectile.move():
                new_projectiles.append(projectile)
            else:
                total_hit_reward += projectile.hit_reward
        self.projectiles = new_projectiles
        # if total_hit_reward > 0:
        #     print("Hit reward: ", total_hit_reward)
        return total_hit_reward
        # new_projectiles = []
        # for projectile in self.projectiles:
        #     if projectile.move():
        #         new_projectiles.append(projectile)
        # self.projectiles = new_projectiles
