import numpy as np
import random
import os
import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame

POP_SIZE = 50
N_GENERATIONS = 500
N_EPISODES = 5
MUTATION_RATE = 0.4
ELITE_FRAC = 0.2
GRID_SIZE = 10
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 3  # type, x, y

title = 'GeneticAgent_50gen'

def preprocess_state(env):
    observation = np.array(env.action_space)
    return observation.reshape(1, GRID_SIZE, GRID_SIZE).flatten()

def decode_action(action_vec):
    # action_vec: [type, x, y] (all in [0,1])
    type_ = int(np.round(action_vec[0]))
    x = int(np.clip(np.round(action_vec[1] * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))
    y = int(np.clip(np.round(action_vec[2] * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))
    return type_, x, y

class Individual:
    def __init__(self):
        # Linear weights: (ACTION_SIZE, STATE_SIZE)
        self.weights = np.random.randn(ACTION_SIZE, STATE_SIZE)
        self.actions = []
        self.fitness = None

    def act(self, state, exploration_std=0.1):
        # state: (STATE_SIZE,)
        action_raw = np.dot(self.weights, state)
        # Add random noise for exploration
        noise = np.random.randn(*action_raw.shape) * exploration_std
        action_noisy = action_raw + noise
        action = 1 / (1 + np.exp(-action_noisy))  # sigmoid to [0,1]
        return decode_action(action)

def evaluate(individual):
    env = TowerDefenseGame()
    total_reward = 0
    for _ in range(N_EPISODES):
        env.reset()
        state = preprocess_state(env)
        episode_reward = 0
        for wave in range(500):
            while env.number_valid_actions():
                for attempt in range(100):
                    type_, i, j = individual.act(state, exploration_std=0.1)
                    tower_type = type_ + 1
                    if env.check_valid_action(i, j, 2):
                        try:
                            individual.actions.append([type_, i, j])
                            env.place_structure_index(i, j, 2, tower_type=tower_type-1)
                            attempt = 100  # Break the attempt loop
                        except:
                            continue
                        break
                else:
                    break
            _, next_state, reward, done, _ = env.step()
            episode_reward += reward
            if done:
                break
            state = preprocess_state(env)
        total_reward += episode_reward
    return total_reward / N_EPISODES

def mutate(weights):
    mutation = np.random.randn(*weights.shape) * MUTATION_RATE
    return weights + mutation

def crossover(parent1, parent2):
    mask = np.random.rand(*parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return child

def get_next_filename(base_name, extension, folder):
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}{i}.{extension}")):
        i += 1
    return os.path.join(folder, f"{base_name}{i}.{extension}")

def main():
    if not os.path.exists(title):
        os.makedirs(title)

    population = [Individual() for _ in range(POP_SIZE)]
    best_rewards = []
    pop_avg_rewards = []

    for gen in range(N_GENERATIONS):
        # Evaluate fitness
        for ind in population:
            ind.fitness = evaluate(ind)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best_rewards.append(population[0].fitness)
        pop_avg = np.mean([ind.fitness for ind in population])
        pop_avg_rewards.append(pop_avg)
        print(f"Generation {gen+1}, Best Avg Reward: {population[0].fitness:.2f}, Population Avg: {pop_avg:.2f}")

        # Elitism
        n_elite = int(ELITE_FRAC * POP_SIZE)
        new_population = population[:n_elite]

        # Crossover and mutation
        while len(new_population) < POP_SIZE:
            parents = random.sample(population[:n_elite], 2)
            child_weights = crossover(parents[0].weights, parents[1].weights)
            child_weights = mutate(child_weights)
            child = Individual()
            child.weights = child_weights
            new_population.append(child)

        population = new_population

    # Plot best rewards and population average per generation
    plt.plot(best_rewards, label='Best Avg Reward')
    plt.plot(pop_avg_rewards, color='red', label='Population Avg Reward')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(f'{title}')
    plt.legend()
    png_filename = get_next_filename(title, 'png', title)
    plt.savefig(png_filename)

    # Save the rewards vector
    txt_filename = get_next_filename(title, 'txt', title)
    with open(txt_filename, 'w') as f:
        for reward in best_rewards:
            f.write(f"{reward:.2f}\n")
    average_title = f"average_{title}"
    txt_filename2 = get_next_filename(average_title, 'txt', title)
    with open(txt_filename2, 'w') as f:
        for reward in pop_avg_rewards:
            f.write(f"{reward:.2f}\n")

    # Save best individual
    best_ind = population[0]
    np.save(os.path.join(title, 'best_genetic_weights.npy'), best_ind.weights)
    print("Best genetic agent saved.")

if __name__ == "__main__":
    main()
