# Random scenario
import random
from Game_env import TowerDefenseGame
env = TowerDefenseGame(game_speed= 60, render=True, training_mode= True, selected_map="map_12.txt")

env.load_map()
def random_place_structures(env):
    while env.number_valid_actions() > 0:
        i = random.randint(0, env.GRID_SIZE - 1)
        j = random.randint(0, env.GRID_SIZE - 1)
        type = random.choice([1, 3])
        print(i, j, type)
        if env.check_valid_action(i, j, type):
            print("valid")
            env.place_structure_index(i, j, type)
            break
        else:
            print("invalid")
            continue

#env.step()
# random_place_structures(env)
# random_place_structures(env)
# random_place_structures(env)
# env.step()
env.place_structure_index(1, 0, 2, tower_type=1)
env.place_structure_index(1, 2, 2, tower_type=1)
total_reward = 0
# env.place_structure_index(1, 2, 2, tower_type=0)
# env.place_structure_index(1, 4, 2, tower_type=0)
# env.place_structure_index(1, 6, 2, tower_type=0)
next_state, next_observation, reward, done, _ = env.step()
total_reward += reward
print(f"Total reward after first step: {total_reward}")
env.place_structure_index(1, 4, 2, tower_type=1)
# env.place_structure_index(1, 10, 2, tower_type=0)
next_state, next_observation, reward, done, _ = env.step()
total_reward += reward
print(f"Total reward after second step: {total_reward}")
env.place_structure_index(1, 6, 2, tower_type=1)
#env.place_structure_index(1, 14, 2, tower_type=0)
next_state, next_observation, reward, done, _ = env.step()
total_reward += reward
print(f"Total reward after third step: {total_reward}")
env.place_structure_index(1, 8, 2, tower_type=1)
#env.place_structure_index(3, 17, 2, tower_type=0)
next_state, next_observation, reward, done, _ = env.step()
total_reward += reward
print(f"Total reward after fourth step: {total_reward}")
env.place_structure_index(4, 8, 2, tower_type=0)
next_state, next_observation, reward, done, _  = env.step()
total_reward += reward
print (total_reward)
# env.place_structure_index(1, 2, 1)
# env.place_structure_index(0, 4, 2)
# env.place_structure_index(2, 4, 1)
# env.place_structure_index(3, 4, 2)
# env.place_structure_index(4, 3, 1)
# env.place_structure_index(5, 1, 2)


env.step()
env.step()
env.step()
env.step()
env.step()
env.step()
env.step()
env.step()

# env.place_structure_index(1, 12, 2)
# env.place_structure_index(1, 14, 2)
# env.step()

# env.place_structure_index(1, 16, 2)
# env.place_structure_index(3, 16, 2)
# env.step()