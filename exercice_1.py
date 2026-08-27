import gymnasium as gym

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")

episodes = 10

for ep in range(episodes):
    
    # Reset environment to start a new episode
    observation, info = env.reset()
    total_reward = 0
    episode_over = False
    print(f"Episode number : {ep}")

    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = env.action_space.sample()  # Random action for now - real agents will be smarter!

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        # print("sample action :", env.action_space.sample())
        # print("observation space shape :", env.observation_space)
        # print("sample observation :", env.observation_space.sample())
        
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")

env.close()
