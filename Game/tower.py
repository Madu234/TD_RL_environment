import pygame
import math

class Projectile:
    def __init__(self, start_pos, target, damage):
        self.x, self.y = start_pos
        self.target = target
        self.damage = damage
        self.speed = 5  # You can adjust this value
        self.color = (0, 0, 0)  # Projectile color - currently black
        self.radius = 5  # Radius of the projectile circle

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
            self.target.hp -= self.damage
            return True  # Indicates that the projectile hit the target

        return False  # Projectile has not hit the target yet

    def draw(self, WIN):
        pygame.draw.circle(WIN, self.color, (int(self.x), int(self.y)), self.radius)

class Tower:
    def __init__(self, pos, cell_size, range, attack_speed):
        self.x = (pos[0]+1) * cell_size
        self.y = (pos[1]+1) * cell_size

        self.range = range
        self.attack_speed = attack_speed
        self.target = None
        self.projectiles = []
        self.color = (255, 0, 0)  # Color of the tower
        self.width = 20  # Width of the tower
        self.height = 20  # Height of the tower
        self.damage = 10
        self.last_shot_time = pygame.time.get_ticks()

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

    def is_in_range(self, enemy):
        return ((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2) ** 0.5 <= self.range

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if self.target and current_time - self.last_shot_time > 1000 / self.attack_speed:
            self.projectiles.append(Projectile((self.x, self.y), self.target, self.damage))
            self.last_shot_time = current_time

    def update_projectiles(self):
        self.projectiles = [projectile for projectile in self.projectiles if not projectile.move()]
