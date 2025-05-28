import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
print (sys.executable)
import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        #self.fc2 = nn.Linear(64,64)
        #self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        #x = torch.sigmoid(self.fc2(x))
        #x = torch.sigmoid(self.fc4(x))
        #x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, episodes):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / episodes
        self.learning_rate = 0.01
        self.batch_size = 8
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to PyTorch tensor
            target_f = self.model(state)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

def preprocess_state(env):
    observation = np.array(env.action_space).flatten()
    return observation

def get_next_filename(base_name, extension, folder):
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}{i}.{extension}")):
        i += 1
    return os.path.join(folder, f"{base_name}{i}.{extension}")

def main():
    env = TowerDefenseGame()
    state_dim = env.GRID_SIZE * env.GRID_SIZE
    action_dim = env.GRID_SIZE * env.GRID_SIZE * 2  # (i, j, type) where type is 1 or 3
    
    episodes = 1000
    title = 'DQN128_1000episodes'
    agent = DQNAgent(state_dim, action_dim, episodes)

    rewards = []
    avg_rewards = []
    rate_episode = 10
    for e in range(episodes):
        env.reset()
        observation = preprocess_state(env)
        total_reward = 0
        episode_reward = 0
        actions = []
        for wave in range(500):
            # while env.waves_available():
            while env.number_valid_actions():
                while True:
                    # Use the model to select an action
                    action = agent.act(observation)
                    i = action // (env.GRID_SIZE * 2)
                    j = (action % (env.GRID_SIZE * 2)) // 2
                    type = 1 if (action % 2) == 0 else 2
                    
                    # Random action
                    # i = random.randint(0, 19)
                    # j = random.randint(0, 19)
                    # type = 2
                    if env.check_valid_action(i, j, 2):
                        try:
                            env.place_structure_index(i, j, 2, tower_type=type-1)
                            actions.append(action)
                        except:
                            continue
                        break
                    else:
                        continue
            
            next_state, next_observation, reward, done, _ = env.step()
            episode_reward += reward
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {episode_reward}, e: {agent.epsilon:.2}")
                break
            next_observation = preprocess_state(env)
        agent.remember(observation, actions, episode_reward, next_observation, done)
        agent.replay()  # Call replay after each step
        observation = next_observation
        total_reward += episode_reward
            
        rewards.append(total_reward)

        # Calculate average reward every 100 episodes
        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward}")

    # Plot the average rewards
    plt.plot(avg_rewards)
    plt.xlabel(f'Average Reward per {rate_episode} Episodes')
    plt.ylabel('Average Reward')
    
    
    
    plt.title(f'{title}')

    # Create a folder with the title name
    if not os.path.exists(title):
        os.makedirs(title)

    # Save the plot
    png_filename = get_next_filename(title, 'png', title)
    plt.savefig(png_filename)
    

    # Save the rewards vector
    txt_filename = get_next_filename(title, 'txt', title)
    with open(txt_filename, 'w') as f:
        for reward in rewards:
            f.write(f"{reward:.2f}\n")
    average_title = f"average_{title}"
    txt_filename2 = get_next_filename(average_title, 'txt',title)
    with open(txt_filename2, 'w') as f:
        for avg_reward in avg_rewards:
            f.write(f"{avg_reward:.2f}\n")

    # Save the trained model
    model_filename = get_next_filename(title, 'pt', title)  # Use 'h5' for TensorFlow/Keras
    torch.save(agent.model.state_dict(), model_filename)  # Use model.save(model_filename) for TensorFlow/Keras

    #plt.show()
    plt.clf()  # Clear the current figure

if __name__ == "__main__":
    main()
    main()
    main()
    main()
    main()