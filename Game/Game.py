import pygame
from enemy import Enemy
from enemy_spawner import EnemySpawner
from calculate_optimal_path import CalculateOptimalPath
from tower import Tower

class TowerDefenseGame:
    def __init__(self):
        pygame.init()

        self.wave_is_on_going = False
        self.WIDTH, self.HEIGHT = 800, 800
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.FPS = 60

        self.WHITE = (255, 255, 255)
        self.GREY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.VIOLET = (127,0,255)

        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        self.to_be_placed = {'tower': 4, 'wall': 0}
        self.enemies = []
        self.enemy_spawner = None  # Make this None initially

        self.towers = []
        self.walls = []

        self.start_point = (0, 0)
        self.end_point = (0, 19)
        self.grid[self.start_point[0]][self.start_point[1]] = "start"
        self.grid[self.end_point[0]][self.end_point[1]] = "finish"



    def draw_grid(self):
        self.WIN.fill(self.WHITE)  # Add this line
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                rect = pygame.Rect(j * self.CELL_SIZE, i * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[i][j] == "tower" and (i != 0 or i != self.CELL_SIZE):
                    pygame.draw.rect(self.WIN, self.RED, rect)
                elif self.grid[i][j] == "wall":
                    pygame.draw.rect(self.WIN, self.BLUE, rect)
                elif self.grid[i][j] == "obstacle":
                    pygame.draw.rect(self.WIN, self.GREY, rect)
                elif self.grid[i][j] == "start" or self.grid[i][j] == "finish":
                    pygame.draw.rect(self.WIN, self.VIOLET, rect)
                else:
                    pygame.draw.rect(self.WIN, self.GREY, rect, 1)

    def draw_enemies(self):
        for enemy in self.enemies:
            enemy.draw(self.WIN)

    def draw_projectiles(self):
        for tower in self.towers:
            for projectile in tower.projectiles:
                projectile.draw(self.WIN)

    def update_enemies(self):
        for enemy in self.enemies:
            enemy.move()

        self.enemies = [enemy for enemy in self.enemies if enemy.is_alive()]
        self.enemies = [enemy for enemy in self.enemies if not (enemy.cell_x, enemy.cell_y) == self.end_point]

        if all(value == 0 for value in self.to_be_placed.values()) and not self.wave_is_on_going:
            path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
            # print (self.end_point)
            optimal_path = path_finder.calculate()
            # Check if optimal_path was not found
            if not optimal_path:
                raise ValueError("Optimal path not found. Ensure that the path can be calculated given the grid, start, and end points.")
            # print("Optimal path:", optimal_path)
            self.enemy_spawner = EnemySpawner(path=optimal_path,enemy_type=2, start_point=self.start_point, end_point=self.end_point, enemy_number=10, enemy_frequency=500, cell_size=self.CELL_SIZE)
            self.wave_is_on_going = True

        if self.wave_is_on_going:
            new_enemy = self.enemy_spawner.spawn()
            if new_enemy is not None:
                self.enemies.append(new_enemy)

    def update_towers(self):
        index = 0
        for tower in self.towers:
            # print(index)
            index = index + 1
            tower.find_target(self.enemies)
            tower.shoot()
            tower.update_projectiles()
            # tower.draw(self.WIN)

    def place_structure_pixels(self, x, y, type):
        i, j = y // self.CELL_SIZE, x // self.CELL_SIZE
        # print(type)
        # print(f"i={i}, j={j}")
        if type == 1 and self.to_be_placed['tower'] > 0:
            if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                for di in range(2):
                    for dj in range(2):
                        self.grid[i + di][j + dj] = "tower"
                self.towers.append(Tower((x, y), self.CELL_SIZE, range=100, attack_speed=4))  # Create a new Tower instance
                self.to_be_placed['tower'] -= 1
        elif type == 3 and self.to_be_placed['wall'] > 0:
            if self.grid[i][j] == "":
                self.grid[i][j] = "wall"
                self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                self.to_be_placed['wall'] -= 1
        # print(self.grid)

    def place_structure_index(self, i, j, type):
        #print(f"place_structure_index called: i={i}, j={j}, type={type}")
        if type == 1 and self.to_be_placed['tower'] > 0:
            if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                for di in range(2):
                    for dj in range(2):
                        self.grid[i + di][j + dj] = "tower"
                self.towers.append(Tower((i, j), self.CELL_SIZE, range=100, attack_speed=4))  # Create a new Tower instance
                self.to_be_placed['tower'] -= 1
        elif type == 3 and self.to_be_placed['wall'] > 0:
            if self.grid[i][j] == "":
                self.grid[i][j] = "wall"
                self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                self.to_be_placed['wall'] -= 1
        else:
            self.grid[i][j] = "obstacle"
        # print(self.grid)
    

    def calculate_reward(self):
        # Assuming you've updated your game logic to calculate
        # optimal_path_length elsewhere (e.g., within update_enemies or similar)
        path_finder = CalculateOptimalPath(self.grid, self.start_point, [self.end_point[0],self.end_point[1]])
        optimal_path_length = len(path_finder.calculate())

        # You can normalize or scale this reward if you'd like
        reward = optimal_path_length

        return reward

    def load_map(self):
        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        file = open("map_empty.txt","r")
        for x_index, line in enumerate(file):
            line = line.split(" ")
            for y_index,position in enumerate(line):
                type = position[0]
                if type == 'S' or type == 's':
                    self.start_point = (x_index, y_index)
                    self.grid[self.start_point[0]][self.start_point[1]] = "start"
                elif type == 'E' or type == 'e':
                    self.end_point = (x_index, y_index)
                    self.grid[self.end_point[0]][self.end_point[1]] = "finish"
                elif type != '0':
                    # print(x_index,y_index)
                    self.place_structure_index(x_index, y_index, int(type))
                    # do nothing
                
            # print(line)


    def main(self):
        clock = pygame.time.Clock()
        run = True
        self.load_map()
        # self.place_structure(1, 1, 3)
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
                            self.towers.append(Tower((j, i), self.CELL_SIZE, range=100, attack_speed=4))  # Create a new Tower instance
                            self.to_be_placed['tower'] -= 1
                    elif event.button == 3 and self.to_be_placed['wall'] > 0:
                        if self.grid[i][j] == "":
                            self.grid[i][j] = "wall"
                            self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                            self.to_be_placed['wall'] -= 1
            self.update_towers()
            self.update_enemies()
            self.draw_grid()
            self.draw_enemies()
            self.draw_projectiles()

            pygame.display.update()

        pygame.quit()


