import pygame
import math
from enemy import Enemy  # Make sure to import your existing Enemy class

class Projectile:
    def __init__(self, x, y, target, speed, damage):
        self.x = x
        self.y = y
        self.target = target
        self.speed = speed
        self.damage = damage

    def move(self):
        if self.target:
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:  # To avoid division by zero
                self.x += self.speed * dx / dist
                self.y += self.speed * dy / dist

            # Check if the projectile has reached the target
            if math.sqrt((self.target.x - self.x) ** 2 + (self.target.y - self.y) ** 2) < self.target.width / 2:
                self.hit_target()

    def draw(self, win):
        pygame.draw.circle(win, (0, 0, 255), (int(self.x), int(self.y)), 5)

    def hit_target(self):
        if self.target:
            self.target.take_damage(self.damage)
            self.target = None  # Remove reference to target

class Tower:
    def __init__(self, x, y, range, color, projectile_speed, projectile_damage,fire_rate):
        self.x = x
        self.y = y
        self.range = range
        self.color = color
        self.target = None
        self.projectiles = []
        self.projectile_speed = projectile_speed
        self.projectile_damage = projectile_damage
        self.fire_rate = fire_rate  # fire rate in terms of frames
        self.fire_cooldown = 0  # cooldown counter to control shooting


    def update_towers(self, enemies):
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        self.find_target(enemies)
        self.shoot()
        self.update_projectiles()
    def draw_towers(self):
        for tower in self.towers:
            tower.draw(self.WIN)

    def find_target(self, enemies):
        for enemy in enemies:
            if self.is_within_range(enemy):
                self.target = enemy
                break
        else:
            self.target = None

    def is_within_range(self, enemy):
        distance = math.sqrt((self.x - enemy.x)**2 + (self.y - enemy.y)**2)
        return distance <= self.range

    def shoot(self):
        if self.target and self.fire_cooldown <= 0:
            self.projectiles.append(Projectile(self.x, self.y, self.target, self.projectile_speed, self.projectile_damage))
            self.fire_cooldown = self.fire_rate

    def update_projectiles(self):
        for projectile in self.projectiles[:]:
            projectile.move()
            if not projectile.target:
                self.projectiles.remove(projectile)