import pygame
from enemy import Enemy
from enemy_spawner import EnemySpawner
from calculate_optimal_path import CalculateOptimalPath
from tower import Tower
import os
import pdb
import time

class TowerDefenseGame:
    def __init__(self, game_speed = 60):
        pygame.init()

        self.wave_is_on_going = False
        self.WIDTH, self.HEIGHT = 800, 800
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.FPS = game_speed
        self.current_optimal_path = []
        self.WHITE = (255, 255, 255)
        self.GREY = (200, 200, 200)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.VIOLET = (127,0,255)

        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.observable_space = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.action_space = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        # Make this a class placable structure
        self.to_be_placed = {'tower': 4, 'wall':0}
        self.structure_dict = {1: "wall", 2: "tower"}
        self.struct_size_array = [1,2]

        self.enemies = []
        self.enemy_spawner = []
        
        self.towers = []
        self.walls = []
        

        self.start_point = (0, 0)
        self.end_point = (0, 19)
        self.grid[self.start_point[0]][self.start_point[1]] = "start"
        self.grid[self.end_point[0]][self.end_point[1]] = "finish"
        # self.action_space = []
        

        self.current_reward = 0
        self.waves = [
            [(1, 5), (2, 0)],  # Wave 1: 10 light enemies
            [(1, 5), (2, 0)],  # Wave 1: 10 light enemies
            #[(1, 10), (2, 0)],  # Wave 2: 10 light, 5 armored
            #[(1, 10), (2, 3)], # Wave 3: 10 light, 10 armored
        ]
        self.current_frame = 0
        self.current_wave_index = 0
        self.active_spawners = []
        self.enemies = []
        self.enemy_spawner = []  # Make this None initially

        self.rewards = 5

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

    def update_enemies(self,current_frame):
        for enemy in self.enemies:
            damage = enemy.move()
            try:
                if damage > 0:
                    self.agent_take_damage(5)
                    # print(self.current_reward)
            except:
                pass
                # Target not yet reached
        for enemy in self.enemies:
            if not enemy.is_alive():
                # print("Enemy died, reward added")
                self.current_reward += 1
        self.enemies = [enemy for enemy in self.enemies if enemy.is_alive()]
        self.enemies = [enemy for enemy in self.enemies if not (enemy.cell_x, enemy.cell_y) == self.end_point]

        if self.wave_is_on_going:
            # try:

            for spawner in self.active_spawners:  
                new_enemy = spawner.spawn(current_frame)
                if new_enemy is not None:
                    self.enemies.append(new_enemy)
                if spawner.get_spawns_left() <= 0:
                    self.active_spawners.remove(spawner)
            if self.active_spawners == [] and self.enemies == []:
                self.wave_is_on_going = False
                # self.to_be_placed['tower'] += 2
            # except:
            #     print("spawinging error")

    def update_towers(self,current_frame):
        index = 0
        for tower in self.towers:
            index = index + 1
            tower.find_target(self.enemies)
            tower.shoot(current_frame)
            self.current_reward += tower.update_projectiles()

    def place_structure_pixels(self, x, y, type):
        i, j = y // self.CELL_SIZE, x // self.CELL_SIZE
        # print(f"i={i}, j={j}")
        if type == 1 and self.to_be_placed['tower'] > 0:
            if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                for di in range(2):
                    for dj in range(2):
                        self.grid[i + di][j + dj] = "tower"
                self.towers.append(Tower((x, y), self.CELL_SIZE, range=100, attack_speed=20,game_FPS = self.FPS))  # Create a new Tower instance
                self.to_be_placed['tower'] -= 1
        elif type == 3 and self.to_be_placed['wall'] > 0:
            if self.grid[i][j] == "":
                self.grid[i][j] = "wall"
                self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                self.to_be_placed['wall'] -= 1
        self.update_action_space()

    def place_structure_index(self, i, j, type):
        #print(f"place_structure_index called: i={i}, j={j}, type={type}")
        if self.check_valid_action(i,j,type) and self.to_be_placed[self.structure_dict[type]]> 0:
            if type == 2 and self.to_be_placed['tower'] > 0:
                if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
                    for di in range(2):
                        for dj in range(2):
                            self.grid[i + di][j + dj] = "tower"
                    self.towers.append(Tower((j, i), self.CELL_SIZE, range=100, attack_speed=20, game_FPS= self.FPS))  # Create a new Tower instance
                    self.to_be_placed['tower'] -= 1
            elif type == 1 and self.to_be_placed['wall'] > 0:
                if self.grid[i][j] == "":
                    self.grid[i][j] = "wall"
                    self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
                    self.to_be_placed['wall'] -= 1
            else:
                self.grid[i][j] = "obstacle"
            self.update_action_space()
            return True
        else:
            return False
    

    def get_reward(self):
        return self.current_reward

    def load_map(self):
        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        # print(os.getcwd())
        try:
            file = open("map_empty.txt","r")
        except:
            # print("exception")
            file = open("C:\\Users\\Madu\\Desktop\\Disertatie\\TD_RL_environment\\Game\\map_empty.txt","r")
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
        self.update_action_space()        
            # print(line)
    
    def update_observable_space(self):
        for row in self.grid:
            for col in row:
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

    def step(self):
        done = False
        info = ""
        self.start_wave()
        self.current_reward = 0
        while self.wave_is_on_going:
            # pygame.event.get()
            self.current_frame += 1
            # clock = pygame.time.Clock()
            # clock.tick(self.FPS)
            self.progress_wave(self.current_frame)
            # self.render()
            # print(f"Frames: {frames}and clock: {clock}")
            # time.sleep(0.1)
        # self.update_action_space()
        # print(f"Frames: {self.current_frame}")
        print("wave over")
        # self.current_wave_index += 1
        if self.current_wave_index == len(self.waves) - 1:
            done = True
            # print("Game over")
        # print(f"Wave no.{self.current_wave_index} is done! Current reward: {self.current_reward}.")

        # for tower in self.towers:
        #     print(f"Shots fired by tower: {tower.shots_fired}")
        return  self.action_space, self.observable_space, self.current_reward, done, info
        
    def start_wave(self):
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
        for enemy_type, quantity in wave:
            self.new_spawner = EnemySpawner(path=self.current_optimal_path, enemy_type=enemy_type, 
                                              start_point=self.start_point, 
                                              end_point=self.end_point, 
                                              enemy_number=quantity, enemy_frequency=30, cell_size=self.CELL_SIZE
                                              ,game_FPS=self.FPS)
            self.active_spawners.append(self.new_spawner)
        self.current_wave_index = self.current_wave_index + 1

        # for row in self.grid:
        #     print(row)
        # for row in self.action_space:
        #     print(row)


    def progress_wave(self,current_frame):
        self.update_towers(current_frame)
        self.update_enemies(current_frame)
        return 
    
    def render(self):
        self.draw_grid()
        self.draw_enemies()
        self.draw_projectiles()
        pygame.display.update()

    def check_valid_action(self, i, j, type):
        # for row in self.grid:
        #     print(row)
        # for row in self.action_space:
        #     print(row)
        shift = 0
        if type == 1:
            shift = 0
        elif type == 2:
            shift = 1
        return (self.action_space[i][j] >> (shift)  & 1)

    def reset(self):
        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.to_be_placed = {'tower': 4, 'wall': 0}
        self.enemies = []
        self.enemy_spawner = []  # Make this None initially

        self.towers = []
        self.walls = []
        self.load_map()
        self.starting_reward = 100
        self.waves = [
            [(1, 5), (2, 0)],            # Wave 1: 10 light enemies
            [(1, 5), (2, 0)],  # Wave 1: 10 light enemies
            #[(1, 10), (2, 5)],  # Wave 2: 10 light, 5 armored
            #[(1, 10), (2, 10)], # Wave 3: 10 light, 10 armored
        ]
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
                            
                            # TO DO: Expereriment with local A*
                            buffer_path = CalculateOptimalPath(buffer_grid,self.start_point,self.end_point).calculate()
                            if buffer_path:
                                self.action_space[row][col] += 2**index
                        else:
                            self.action_space[row][col] += 2**index
        # for row in self.grid:
        #     print(row)
        # for row in self.action_space:
        #     print(row)

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


    # def main(self):
    #     clock = pygame.time.Clock()
    #     run = True
    #     self.load_map()
    #     while run:
    #         clock.tick(self.FPS)
    #         if (self.wave_is_on_going == False):
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     run = False
    #                 if event.type == pygame.MOUSEBUTTONDOWN:
    #                     x, y = pygame.mouse.get_pos()
    #                     i, j = y // self.CELL_SIZE, x // self.CELL_SIZE
    #                     if event.button == 1 and self.to_be_placed['tower'] > 0 and i < self.GRID_SIZE-1 and j < self.GRID_SIZE-1:
    #                         if all(self.grid[i + di][j + dj] == "" for di in range(2) for dj in range(2)):
    #                             for di in range(2):
    #                                 for dj in range(2):
    #                                     self.grid[i + di][j + dj] = "tower"
    #                             self.towers.append(Tower((j, i), self.CELL_SIZE, range=100,
    #                                                       attack_speed=1))  # Create a new Tower instance
    #                             self.to_be_placed['tower'] -= 1
    #                     elif event.button == 3 and self.to_be_placed['wall'] > 0:
    #                         if self.grid[i][j] == "":
    #                             self.grid[i][j] = "wall"
    #                             self.walls.append((j * self.CELL_SIZE, i * self.CELL_SIZE))  # add this line
    #                             self.to_be_placed['wall'] -= 1

    #             if all(value == 0 for value in self.to_be_placed.values()):        
    #                 self.start_wave()
                
    #         self.progress_wave()
    #         self.render()

            # self.progress_wave()
            # self.render()
            

    #    pygame.quit()


