import gymnasium as gym

env = gym.make("CartPole-v1")   # create a simple test env
obs, info = env.reset()
print("Observation:", obs)
print("Info:", info)

env.close()
