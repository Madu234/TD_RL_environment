import gym

env = TowerDefenseGymEnv()
observation = env.reset()

for _ in range(1000): 
    action = env.action_space.sample()  # Replace with your agent's decision logic
    observation, reward, done, info = env.step(action)
    env.render()  # If you implement rendering in your env
    if done:
        observation = env.reset()
