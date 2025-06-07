import pygame
from enemy import Enemy
from enemy_spawner import EnemySpawner
from calculate_optimal_path import CalculateOptimalPath
from tower import Tower
import os
import pdb
import time
import random
class TowerDefenseGame:
    def __init__(self, game_speed = 60, render = False, training_mode = True, selected_map = ""):
        pygame.init()

        self.wave_is_on_going = False
        # self.WIDTH, self.HEIGHT = 800, 800
        self.GRID_SIZE = 10
        # 40 CELL SIZE
        self.CELL_SIZE = 40
        self.WIDTH = self.CELL_SIZE * self.GRID_SIZE
        self.HEIGHT = self.CELL_SIZE * self.GRID_SIZE
        self.render_flag = render
        self.FPS = game_speed
        self.current_optimal_path = []
        self.WHITE = (255, 255, 255)
        self.GREY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.VIOLET = (127,0,255)
        self.training_mode = training_mode
        self.selected_map = selected_map

        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.observable_space = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.action_space = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        
        self.structure_dict = {1: "wall", 2: "tower", 3: "obstacle"}
        self.struct_size_array = [1,2]

        self.enemies = []
        self.enemy_spawner = []
        
        self.towers = []
        self.walls = []
        #self.spawner_delay = 0

        self.start_point = (0, 0)
        self.end_point = (0, 9)
        self.grid[self.start_point[0]][self.start_point[1]] = "start"
        self.grid[self.end_point[0]][self.end_point[1]] = "finish"
        # self.action_space = []
        

        self.current_reward = 0
        self.initial_to_be_placed ={'tower': 2, 'wall':0, 'obstacle':0}
        self.initial_waves =[
            # enemy_type : 1 - light, 2 - armored
            # enemy_type, quantity, enemy_frequency , delay
            [(1, 1, 30, 0), (2, 0, 30, 0)],
            [(1, 2, 30, 0), (2, 0, 30, 0)],
            [(1, 3, 30, 0), (2, 0, 30, 0)],
            [(1, 8, 30, 0), (2, 1, 30, 0)],
            [(1, 10, 15, 0), (2, 4, 30, 0)],
            #[(1, 4), (2, 0)],
            #[(1, 5), (2, 0)],
            #[(1, 5), (2, 0)],
            #[(1, 6), (2, 0)],
            #[(1, 6), (2, 0)],
        ]
        # Make this a class placable structure
        self.to_be_placed = self.initial_to_be_placed
        self.waves = self.initial_waves
        self.current_frame = 0
        self.current_wave_index = 0
        self.active_spawners = []
        self.enemies = []
        self.enemy_spawner = []  # Make this None initially
        
    def number_valid_actions(self):
        sum = 0
        for value in self.to_be_placed.values():
            sum += value
        return sum

    def get_remaining_moves(self):
        return sum(list(self.to_be_placed.values()))

    def agent_take_damage(self, damage):
        self.current_reward = self.current_reward - 1


    def draw_grid(self):
        self.WIN.fill(self.WHITE)  # Add this line
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                size_rect = pygame.Rect(j * self.CELL_SIZE, i * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                # Draw contor line start with i=0, increment j when detected a tower drawn a slighty blackish square and over it the normal color
                # there will 10-20 pixel of blackish color
                if self.grid[i][j] == "tower" and (i != 0 or i != self.CELL_SIZE):
                    color = self.RED
                    for tower in self.towers:
                        if j == tower.i and i == tower.j:
                            color = tower.get_color()
                        if j-1 == tower.i and i == tower.j:
                            color = tower.get_color()
                        if j == tower.i and i-1 == tower.j:
                            color = tower.get_color()
                        if j-1 == tower.i and i-1 == tower.j:
                            color = tower.get_color()
                    pygame.draw.rect(self.WIN, color, size_rect)
                elif self.grid[i][j] == "wall":
                    pygame.draw.rect(self.WIN, self.BLUE, size_rect)
                elif self.grid[i][j] == "obstacle":
                    pygame.draw.rect(self.WIN, self.GREY, size_rect)
                elif self.grid[i][j] == "start" or self.grid[i][j] == "finish":
                    pygame.draw.rect(self.WIN, self.VIOLET, size_rect)
                else:
                    pygame.draw.rect(self.WIN, self.GREY, size_rect, 1)

    def draw_enemies(self):
        for enemy in self.enemies:
            enemy.draw(self.WIN)

    def draw_projectiles(self):
        for tower in self.towers:
            for projectile in tower.projectiles:
                projectile.draw(self.WIN)

    def update_enemies(self,current_frame):
        for enemy in self.enemies:
            self.current_reward -= enemy.move()
        for enemy in self.enemies:
            if not enemy.is_alive():
                #print("Enemy died, reward added")
                self.current_reward += 1
        self.enemies = [enemy for enemy in self.enemies if enemy.is_alive()]
        #self.enemies = [enemy for enemy in self.enemies if not (enemy.cell_x, enemy.cell_y) == self.end_point]
        new_enemies = []
        for enemy in self.enemies:
            if (enemy.cell_x, enemy.cell_y) != self.end_point:
                
                new_enemies.append(enemy)
        self.enemies = new_enemies

        if self.wave_is_on_going:
            for spawner in self.active_spawners:  
                new_enemy = spawner.spawn(current_frame)
                if new_enemy is not None:
                    self.enemies.append(new_enemy)
                if spawner.get_spawns_left() <= 0:
                    self.active_spawners.remove(spawner)
            if self.active_spawners == [] and self.enemies == []:
                self.wave_is_on_going = False
                self.to_be_placed['tower'] += 1

    def update_towers(self,current_frame):
        for tower in self.towers:
            tower.find_target(self.enemies)
            tower.shoot(current_frame)
            tower.update_projectiles()
            # self.current_reward += tower.update_projectiles()

    def place_structure_pixels(self, x, y, type):
        i, j = y // self.CELL_SIZE, x // self.CELL_SIZE
        # print(f"i={i}, j={j}")
        if type == 1 and self.to_be_placed['tower'] > 0:
            if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                for di in range(2):
                    for dj in range(2):
                        self.grid[i + di][j + dj] = "tower"
                self.towers.append(Tower((x, y), self.CELL_SIZE, range=100, Reload_time=20))  # Create a new Tower instance
                self.to_be_placed['tower'] -= 1
        elif type == 3 and self.to_be_placed['wall'] > 0:
            if self.grid[i][j] == "":
                self.grid[i][j] = "wall"
                self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                self.to_be_placed['wall'] -= 1
        self.update_action_space()

    def place_structure_index(self, i, j, type, loading_stage = False, tower_type = 1):
        #print(f"place_structure_index called: i={i}, j={j}, type={type}")
        if self.check_valid_action(i,j,type) and self.to_be_placed[self.structure_dict[type]]> 0:
            if type == 2 and self.to_be_placed['tower'] > 0:
                if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                    for di in range(2):
                        for dj in range(2):
                            self.grid[i + di][j + dj] = "tower"
                    self.towers.append(Tower((j, i), self.CELL_SIZE, range=100, Reload_time=20, type = tower_type))  # Create a new Tower instance
                    self.to_be_placed['tower'] -= 1
            elif type == 1 and self.to_be_placed['wall'] > 0:
                if self.grid[i][j] == "":
                    self.grid[i][j] = "wall"
                    self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                    self.to_be_placed['wall'] -= 1
            elif type == 3 and self.to_be_placed['obstacle'] > 0:
                self.grid[i][j] = "obstacle"
                self.to_be_placed['obstacle'] -= 1
            if not loading_stage:
                self.update_action_space()
            return True
        else:
            return False
    

    def get_reward(self):
        return self.current_reward

    def load_map(self):
        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        # Determine folder for maps
        base_dir = os.path.dirname(__file__)
        if self.training_mode:
            folder = os.path.join(base_dir, "Maps", "Training")
        else:
            folder = os.path.join(base_dir, "Maps", "Evaluation")
        map_files = [f for f in os.listdir(folder) if f.startswith("map_") and f.endswith(".txt")]
        # Pick a random map unless selected_map is set
        if self.selected_map:
            map_file = self.selected_map
        else:
            map_file = random.choice(map_files)
        try:
            file = open(os.path.join(folder, map_file), "r")
        except Exception as e:
            print(f"Could not open map file: {os.path.join(folder, map_file)}, error: {e}")
            # fallback to empty map
            file = open(os.path.join(base_dir, "map_empty.txt"), "r")
        for x_index, line in enumerate(file):
            line = line.split(" ")
            for y_index, position in enumerate(line):
                type = position[0]
                if type == 'S' or type == 's':
                    self.start_point = (x_index, y_index)
                    self.grid[self.start_point[0]][self.start_point[1]] = "start"
                elif type == 'E' or type == 'e':
                    self.end_point = (x_index, y_index)
                    self.grid[self.end_point[0]][self.end_point[1]] = "finish"
                elif type != '0':
                    if type == '3':
                        self.to_be_placed['obstacle'] += 1
                    self.place_structure_index(x_index, y_index, int(type), loading_stage=True)
        self.update_action_space()

    def update_observable_space(self):
        for row in self.grid:
            for col in row:
                # change to include type 2 of tower
                if self.grid[row][col] == "tower":
                    self.observable_space[row][col] = 1
                elif self.grid[row][col] == "wall":
                    self.observable_space[row][col] = 3
                elif self.grid[row][col] == "obstacle":
                    self.observable_space[row][col] = 3
                elif self.grid[row][col] == "start":
                    self.observable_space[row][col] = 4
                elif self.grid[row][col] == "finish":
                    self.observable_space[row][col] = 4
                else:
                    self.observable_space[row][col] = 0
        # add future wave incoming 

    def step(self):
        self.wave_is_on_going = True
        done = False
        info = ""
        self.start_wave()
        self.current_reward = 0
        while self.wave_is_on_going:
            self.current_frame += 1
            self.progress_wave(self.current_frame)
            if self.render_flag:
                self.render()
        for tower in self.towers:
            tower.aging()
            #print(f"tower aged {self.current_wave_index}"
        if self.current_wave_index == len(self.waves):
            done = True
        # self.render_flag = False
        return  self.action_space, self.observable_space, self.current_reward, done, info
    
    def start_wave(self):
        #self.spawner_delay =0
        
        path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
        # print (self.end_point)
        self.current_optimal_path = path_finder.calculate()
        # Check if optimal_path was not found
        if not self.current_optimal_path:
            raise ValueError("Optimal path not found. Ensure that the path can be calculated given the grid, start, and end points.")
        self.wave_is_on_going = True
        if self.current_wave_index >= len(self.waves):
            return
        wave = self.waves[self.current_wave_index]
        for enemy_type, quantity, enemy_frequency, delay in wave:
            self.new_spawner = EnemySpawner(path=self.current_optimal_path, enemy_type=enemy_type, 
                                              start_point=self.start_point, 
                                              end_point=self.end_point, 
                                              enemy_number=quantity, enemy_frequency=enemy_frequency, cell_size=self.CELL_SIZE
                                              ,game_FPS=self.FPS, delay = delay)
            #self.spawner_delay +=10
            self.active_spawners.append(self.new_spawner)
        self.current_wave_index = self.current_wave_index + 1
    def are_waves_left():
        return 
    def progress_wave(self,current_frame):
        self.update_towers(current_frame)
        self.update_enemies(current_frame)
        if self.render_flag:
            time.sleep(1/self.FPS)
        return 
    
    def draw_optimal_path(self, path):
        if len(path) > 1:
            for i in range(len(path) - 1):
                start_pos = (path[i][0] * self.CELL_SIZE + self.CELL_SIZE // 2, path[i][1] * self.CELL_SIZE + self.CELL_SIZE // 2)
                end_pos = (path[i + 1][0] * self.CELL_SIZE + self.CELL_SIZE // 2, path[i + 1][1] * self.CELL_SIZE + self.CELL_SIZE // 2)
                pygame.draw.line(self.WIN, (139, 69, 19), start_pos, end_pos, 5)  # Brown color

    def render(self):
        self.render_flag = True
        self.draw_grid()
        self.draw_optimal_path(self.current_optimal_path)  # Assuming self.current_optimal_path contains the optimal path
        self.draw_projectiles()
        self.draw_enemies()
        pygame.display.update()

    def check_valid_action(self, i, j, type):
        shift = 0
        if type == 1:
            shift = 0
        elif type == 2:
            shift = 1
        elif type == 3:
            return True
        return (self.action_space[i][j] >> (shift)  & 1)

    def reset(self):
        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.to_be_placed = self.initial_to_be_placed
        self.enemies = []
        self.enemy_spawner = []  # Make this None initially

        self.towers = []
        self.walls = []
        self.load_map()
        self.starting_reward = 100
        self.waves = self.initial_waves
        self.current_wave_index = 0
        self.active_spawners = []
        self.enemies = []
        self.enemy_spawner = []  # Make this None initially

    def interferes_with_optimal_path(self,i,j,structure_size):
        for j_path, i_path in self.current_optimal_path:
            if (i_path >= i and i_path <= i + structure_size - 1) \
            and (j_path >= j and j_path <= j + structure_size - 1):
                return True
        return False

    def update_action_space(self):
        rows = len(self.grid)
        cols = len(self.grid[0])
        self.action_space = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        # result = [[0 for _ in range(cols)] for _ in range(rows)]  # Initialize result grid
        path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
        # print (self.end_point)
        self.current_optimal_path = path_finder.calculate()
        for row in range(rows):
            for col in range(cols):
                for index, structure_size in enumerate(self.struct_size_array):
                    if self.place_empty(row, col, structure_size):
                        if self.interferes_with_optimal_path(row,col,structure_size):

                            #buffer_grid = self.grid
                            buffer_grid = [[self.grid[j][i] for i in range(self.GRID_SIZE)] for j in range(self.GRID_SIZE)]
                            for i_struct in range(structure_size):
                                for j_struct in range(structure_size):
                                    buffer_grid[row+i_struct][col+j_struct]= 1
                            
                            # TO DO: Experiment with local A*
                            buffer_path = CalculateOptimalPath(buffer_grid,self.start_point,self.end_point).calculate()
                            if buffer_path:
                                self.action_space[row][col] += 2**index
                        else:
                            self.action_space[row][col] += 2**index

    def place_empty(self, row, col, structure_size):
        if structure_size == 1:  # 1x1 structure
            return self.grid[row][col] == ''
        elif structure_size == 2:  # 2x2 structure
            return (
                self.pos_is_valid(self.grid, row, col) 
                and self.pos_is_valid(self.grid, row + 1, col) 
                and self.pos_is_valid(self.grid, row, col + 1) 
                and self.pos_is_valid(self.grid, row + 1, col + 1)
                and self.grid[row][col] == ''
                and self.grid[row + 1][col] == ''
                and self.grid[row][col + 1] == ''
                and self.grid[row + 1][col + 1] == ''
        )

    @staticmethod
    def pos_is_valid(grid, row, col):
            return 0 <= row < len(grid) and 0 <= col < len(grid[0])

    def close(self):
        self.render_flag = False
        pygame.quit()


    def main(self):
        # clock = pygame.time.Clock()
        run = True
        self.load_map()
        while run:
            # clock.tick(self.FPS)
            if (self.wave_is_on_going == False):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        i, j = y // self.CELL_SIZE, x // self.CELL_SIZE
                        type = 3
                        mouse_left_click = 1
                        mouse_right_click = 3
                        if event.button == mouse_left_click:
                            type = 2
                            #type = self.structure_dict.keys([self.structure_dict.values().index("tower")])
                        elif event.button == mouse_right_click:
                            type = 1
                            #type = self.structure_dict.values().index("wall")
                        if self.check_valid_action(i,j,type) and self.to_be_placed[self.structure_dict[type]]> 0:
                            if type == 2 and self.to_be_placed['tower'] > 0:
                                if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                                    for di in range(2):
                                        for dj in range(2):
                                            self.grid[i + di][j + dj] = "tower"
                                    self.towers.append(Tower((j, i), self.CELL_SIZE, range=100, Reload_time=20))  # Create a new Tower instance
                                    self.to_be_placed['tower'] -= 1
                            elif type == 1 and self.to_be_placed['wall'] > 0:
                                if self.grid[i][j] == "":
                                    self.grid[i][j] = "wall"
                                    self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                                    self.to_be_placed['wall'] -= 1
                            elif type == 3 and self.to_be_placed['obstacle'] > 0:
                                self.grid[i][j] = "obstacle"
                                self.to_be_placed['obstacle'] -= 1
                            path_finder = CalculateOptimalPath(self.grid, self.start_point, self.end_point)
                            # print (self.end_point)
                            self.current_optimal_path = path_finder.calculate()
                            # if not loading_stage:
                            #     self.update_action_space()
                            # return True

                if all(value == 0 for value in self.to_be_placed.values()):  
                    print("Wave started")      
                    self.step()
                else:
                    self.render()
                    
                
            # self.progress_wave()


            # self.progress_wave()
            # self.render()
            

    #    pygame.quit()
