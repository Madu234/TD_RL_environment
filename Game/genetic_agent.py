import numpy as np
import random
import os
import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame

POP_SIZE = 50
N_GENERATIONS = 100
N_EPISODES = 5
MUTATION_RATE = 0.1
ELITE_FRAC = 0.2
GRID_SIZE = 10
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 3  # type, x, y

title = 'GeneticAgent_50gen'

def preprocess_state(env):
    # Map grid string values to integers
    mapping = {'': 0, 'wall': 1, 'tower': 2, 'start': 3, 'end': 4}
    grid = np.array([[mapping.get(cell, 0) for cell in row] for row in env.grid], dtype=np.float32)
    return grid.flatten()

def get_valid_actions(env):
    # Return a list of all valid (type, x, y) actions
    valid_actions = []
    for i in range(env.GRID_SIZE):
        for j in range(env.GRID_SIZE):
            for t in [0, 1]:  # type 0 or 1 (maps to tower_type 1 or 2)
                if env.check_valid_action(i, j, 2):
                    valid_actions.append((t, i, j))
    return valid_actions

class Individual:
    def __init__(self):
        # Linear weights: (ACTION_SIZE, STATE_SIZE)
        self.weights = np.random.randn(ACTION_SIZE, STATE_SIZE).astype(np.float32)
        self.actions = []
        self.fitness = None

    def act(self, state, valid_actions, exploration_std=1, random_action_prob=0.1):
        if not valid_actions:
            return None
        if np.random.rand() < random_action_prob:
            return random.choice(valid_actions)
        # Score each valid action using the individual's weights
        state = state.astype(np.float32)  # Ensure float dtype
        scores = []
        ws = np.dot(self.weights, state)  # shape (3,)
        for t, i, j in valid_actions:
            action_vec = np.array([t, i / (GRID_SIZE - 1), j / (GRID_SIZE - 1)], dtype=np.float32)
            score = float(np.dot(ws, action_vec))
            # Add exploration noise
            score += np.random.randn() * exploration_std
            scores.append(score)
        best_idx = np.argmax(scores)
        return valid_actions[best_idx]

def evaluate(individual):
    env = TowerDefenseGame()
    total_reward = 0
    for _ in range(N_EPISODES):
        env.reset()
        state = preprocess_state(env)
        episode_reward = 0
        for wave in range(500):
            while env.number_valid_actions():
                valid_actions = get_valid_actions(env)
                if not valid_actions:
                    break
                for attempt in range(100):
                    action = individual.act(state, valid_actions, exploration_std=0.1, random_action_prob=0.1)
                    if action is None:
                        break
                    type_, i, j = action
                    tower_type = type_ + 1
                    if env.check_valid_action(i, j, 2):
                        try:
                            individual.actions.append([type_, i, j])
                            env.place_structure_index(i, j, 2, tower_type=tower_type-1)
                            break
                        except:
                            continue
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
