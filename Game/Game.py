import pygame
from enemy import Enemy
from tower import Tower
from enemy_spawner import EnemySpawner
from calculate_optimal_path import CalculateOptimalPath
class TowerDefenseGame:
    def __init__(self):
        pygame.init()

        self.wave_is_on_going = False
        self.WIDTH, self.HEIGHT = 800, 800
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.enemy_health = 100
        self.FPS = 60

        self.WHITE = (255, 255, 255)
        self.GREY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        self.to_be_placed = {'tower': 4, 'wall': 10}
        self.enemies = []
        self.enemy_spawner = None  # Make this None initially

        self.towers = []
        self.walls = []

        self.start_point = (0, 0)
        self.end_point = (0, 19)


    def draw_grid(self):
        self.WIN.fill(self.WHITE)  # Add this line
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                rect = pygame.Rect(j * self.CELL_SIZE, i * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[i][j] == "tower" and (i != 0 or i != self.CELL_SIZE):
                    pygame.draw.rect(self.WIN, self.RED, rect)
                elif self.grid[i][j] == "wall":
                    pygame.draw.rect(self.WIN, self.BLUE, rect)
                else:
                    pygame.draw.rect(self.WIN, self.GREY, rect, 1)

    def draw_enemies(self):
        for enemy in self.enemies:
            enemy.draw(self.WIN)

    def update_enemies(self):
        for enemy in self.enemies:
            enemy.move()
            if enemy.is_dead():
                self.enemies.remove(enemy)
                continue  # Skip the rest of the loop for this enemy

        self.enemies = [enemy for enemy in self.enemies if not (enemy.cell_x, enemy.cell_y) == self.end_point]

        if all(value == 0 for value in self.to_be_placed.values()) and not self.wave_is_on_going:
            path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
            optimal_path = path_finder.calculate()
            print("Optimal path:", optimal_path)
            self.enemy_spawner = EnemySpawner(path=optimal_path,enemy_type=Enemy, start_point=self.start_point,
                                              end_point=self.end_point, enemy_number=10, enemy_frequency=500,
                                              cell_size=self.CELL_SIZE, initial_health=self.enemy_health)
            self.wave_is_on_going = True

        if self.wave_is_on_going:
            new_enemy = self.enemy_spawner.spawn()
            if new_enemy is not None:
                self.enemies.append(new_enemy)

    def update_towers_game(self):
        for tower in self.towers:
            tower.update_towers(self.enemies)

    def draw_towers(self):
        for tower in self.towers:
            # Draw the tower base (assuming it's a simple rectangle here)
            pygame.draw.rect(self.WIN, tower.color, (
            tower.x - self.CELL_SIZE // 2, tower.y - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE))
            # Draw the tower's projectiles
            for projectile in tower.projectiles:
                projectile.draw(self.WIN)

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
                    if event.button == 1 and self.to_be_placed['tower'] > 0:
                        if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                            for di in range(2):
                                for dj in range(2):
                                    self.grid[i + di][j + dj] = "tower"
                            tower_x = j * self.CELL_SIZE + self.CELL_SIZE // 2
                            tower_y = i * self.CELL_SIZE + self.CELL_SIZE // 2
                            new_tower = Tower(tower_x, tower_y, range=100, color=self.RED,
                                              projectile_speed=5, projectile_damage=10, fire_rate=10)  # Adjust parameters as needed
                            self.towers.append(new_tower)
                            self.to_be_placed['tower'] -= 1
                    elif event.button == 3 and self.to_be_placed['wall'] > 0:
                        if self.grid[i][j] == "":
                            self.grid[i][j] = "wall"
                            self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                            self.to_be_placed['wall'] -= 1
                    if not CalculateOptimalPath(self.grid, self.start_point, self.end_point).calculate():
                        print("Not a valid path")
                        return 0
            self.update_towers_game()  # Make sure to call this in your main loop
            self.update_enemies()
            self.draw_grid()
            self.draw_enemies()
            self.draw_towers()

            pygame.display.update()

        pygame.quit()

