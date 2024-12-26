import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
print (sys.executable)
# import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
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
            self.epsilon *= self.epsilon_decay

def preprocess_state(env):
    observation = np.array(env.observable_space).flatten()
    return observation

def main():
    env = TowerDefenseGame()
    state_dim = env.GRID_SIZE * env.GRID_SIZE
    action_dim = env.GRID_SIZE * env.GRID_SIZE * 2  # (i, j, type) where type is 1 or 3
    agent = DQNAgent(state_dim, action_dim)
    episodes = 10000
    rewards = []
    avg_rewards = []

    for e in range(episodes):
        env.reset()
        observation = preprocess_state(env)
        total_reward = 0
        for time in range(500):
            while env.number_valid_actions():
                while True:
                    action = agent.act(observation)
                    i = action // (env.GRID_SIZE * 2)
                    j = (action % (env.GRID_SIZE * 2)) // 2
                    type = 1 if (action % 2) == 0 else 2
                    if env.check_valid_action(i, j, type):
                        try:
                            env.place_structure_index(i, j, type)
                        except:
                            continue
                        break
                    else:
                        continue
            next_state, next_observation, reward, done, _ = env.step()
            next_observation = preprocess_state(env)
            agent.remember(observation, action, reward, next_observation, done)
            agent.replay()  # Call replay after each step
            observation = next_observation
            total_reward += reward
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {total_reward}, e: {agent.epsilon:.2}")
                break
        rewards.append(total_reward)

        # Calculate average reward every 100 episodes
        if (e + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward}")

    # Plot the average rewards
    # plt.plot(avg_rewards)
    # plt.xlabel('Episode (in hundreds)')
    # plt.ylabel('Average Reward')
    # plt.title('Average Reward per 100 Episodes')
    # plt.savefig('average_reward.png')
    # plt.show()

if __name__ == "__main__":
    main()