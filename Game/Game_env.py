import pygame
from enemy import Enemy
from enemy_spawner import EnemySpawner
from calculate_optimal_path import CalculateOptimalPath
from tower import Tower
import pdb

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
        self.enemy_spawner = []  # Make this None initially

        self.towers = []
        self.walls = []
        self.agent_life = 100

        self.start_point = (0, 0)
        self.end_point = (0, 19)
        self.grid[self.start_point[0]][self.start_point[1]] = "start"
        self.grid[self.end_point[0]][self.end_point[1]] = "finish"

        self.waves = [
            [(1, 2)],  # Wave 1: 10 light enemies
            [(1, 10), (2, 5)],  # Wave 2: 10 light, 5 armored
            [(1, 10), (2, 10)], # Wave 3: 10 light, 10 armored
        ]
        self.current_wave_index = 0
        self.active_spawners = []


        
    def agent_take_damage(self, damage):
        self.agent_life = self.agent_life - damage


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
            damage = enemy.move()
            try:
                if damage > 0:
                    self.agent_life = self.agent_life - damage
                    print(self.agent_life)
            except:
                pass
                # Target not yet reached

        self.enemies = [enemy for enemy in self.enemies if enemy.is_alive()]
        self.enemies = [enemy for enemy in self.enemies if not (enemy.cell_x, enemy.cell_y) == self.end_point]

        # if all(value == 0 for value in self.to_be_placed.values()) and not self.wave_is_on_going:
        #     path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
        #     # print (self.end_point)
        #     optimal_path = path_finder.calculate()
        #     # Check if optimal_path was not found
        #     if not optimal_path:
        #         raise ValueError("Optimal path not found. Ensure that the path can be calculated given the grid, start, and end points.")
        #     # I need to implement wave system, and I need to find a data structure that is matching my needs.
            
        #     self.start_wave(optimal_path)
        #     self.wave_is_on_going = True
        # print(self.wave_is_on_going)
        # pdb.set_trace()
        if self.wave_is_on_going:
            try:
                for spawner in self.active_spawners:  
                    new_enemy = spawner.spawn()
                    if new_enemy is not None:
                        #pdb.set_trace()
                        self.enemies.append(new_enemy)
                    if spawner.get_spawns_left() <= 0:
                        print("removed")
                        #pdb.set_trace()
                        self.active_spawners.remove(spawner)
                if self.active_spawners == [] and self.enemies == []:
                    self.wave_is_on_going = False
                    self.to_be_placed['tower'] += 2
            except:
                print("error")

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

    def step(self):
        # Start a wave and finish it...
        # start_wave(1)
        # is_ongoing_wave = true
        # while is_ongoing_wave:
        #   is_ongoing_wave = progress_wave(1)
        # 
        # return obs , reward, done

        # obs : grid of current maze
        # reward : enemies past, damage dealt, maze lenght
        # done : player dead or all waves done

        return 
        
    def start_wave(self):
        path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
        # print (self.end_point)
        optimal_path = path_finder.calculate()
        # Check if optimal_path was not found
        if not optimal_path:
            raise ValueError("Optimal path not found. Ensure that the path can be calculated given the grid, start, and end points.")
        # I need to implement wave system, and I need to find a data structure that is matching my needs.
        
        #self.start_wave(optimal_path)
        self.wave_is_on_going = True
        wave = self.waves[self.current_wave_index]
        for enemy_type, quantity in wave:
            self.new_spawner = EnemySpawner(path=optimal_path, enemy_type=enemy_type, 
                                              start_point=self.start_point, 
                                              end_point=self.end_point, 
                                              enemy_number=quantity, enemy_frequency=500, cell_size=self.CELL_SIZE)
            self.active_spawners.append(self.new_spawner)
        # print(self.active_spawners)
        self.current_wave_index = self.current_wave_index + 1


    def progress_wave(self):
        self.update_towers()
        self.update_enemies()
        # update enemies
        # update towers
        # return false if no enemy are alive or left to be spawn
        return 
    
    def render(self):
        self.draw_grid()
        self.draw_enemies()
        self.draw_projectiles()

    def check_valid_action():
        # if optimal_path:
        #   return True
        # else:
        #   remove_action_from_space_action
        #   return False
        return 


    def main(self):
        clock = pygame.time.Clock()
        run = True
        self.load_map()
        # self.place_structure(1, 1, 3)
        while run:
            clock.tick(self.FPS)
            #print("update1")
            # if (self.wave_is_on_going == True):
                #print("update2")
                # self.progress_wave()
                # continue
            if (self.wave_is_on_going == False):
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

                if all(value == 0 for value in self.to_be_placed.values()):        
                    self.start_wave()
                
            self.progress_wave()
            self.render()

            # self.progress_wave()
            # self.render()
            pygame.display.update()

        pygame.quit()


