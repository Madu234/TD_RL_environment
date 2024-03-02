import gym
import stable_baselines3  # Import this
import numpy as np
from Game import TowerDefenseGame  # Import your game class

class TowerDefenseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = TowerDefenseGame()

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=255,  # Adjust based on your grid encoding
                                                shape=(self.game.HEIGHT, self.game.WIDTH, 3),
                                                dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([5, self.game.GRID_SIZE, self.game.GRID_SIZE])  # Discrete actions spaces

    def step(self, action):
        # 1. Unpack the action
        action_type, x, y = action

        # 2. Execute action in the game
        if action_type == 0:  # No action
            pass
        elif action_type == 1:  # Place tower
            self.game.place_structure(x, y, 'tower')
        elif action_type == 2:  # Place wall
            self.game.place_structure(x, y, 'wall')
        # ... add more actions if needed

        # 3. Update game state
        self.game.main()  # Update game logic

        # 4. Construct observation, reward, done, info
        observation = self.get_observation()
        reward = self.calculate_reward()
        done = self.check_game_over()
        info = {}  # Additional info if needed

        return observation, reward, done, info

    def reset(self):
        self.game = TowerDefenseGame()  # Reset game to initial state
        return self.get_observation()

    def render(self, mode="human"):
        # Update your game rendering here, if you wish to visualize
        self.game.draw_grid()
        self.game.draw_enemies()
        self.game.draw_projectiles()
        pygame.display.update()

    def get_observation(self):
        # Create an RGB image representing the current grid state
        # ... your encoding logic here
        return encoded_grid

    def calculate_reward(self):
        # Design your reward function (e.g., negative reward if enemies reach the end, etc.)
        # ... your reward logic here
        return reward

    def check_game_over(self):
        # Check if game is over (win or lose condition)
        # ... your game over logic here
        return is_game_over

def train_model():
    env = TowerDefenseEnv()  # Create your environment
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)

    # Training loop
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("tower_defense_agent")

    # Evaluate the model (add this for performance assessment)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward}")

def evaluate_policy(model, env, n_eval_episodes=10):
    """
    Evaluates a policy (RL model) over a number of episodes

    Args:
        model: The RL model to evaluate
        env:  The Gym environment
        n_eval_episodes: Number of episodes to run for evaluation

    Returns:
        mean_episode_reward: Average reward across evaluation episodes.
        std_episode_reward: Standard deviation of rewards across episodes.
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs)  # Get action from the model
            obs, reward, done, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

    mean_episode_reward = np.mean(episode_rewards)
    std_episode_reward = np.std(episode_rewards)

    return mean_episode_reward, std_episode_reward

if __name__ == "__main__":
    train_model()
