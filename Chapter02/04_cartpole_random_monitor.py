import gymnasium as gym
from gymnasium.wrappers import RecordVideo

if __name__ == "__main__":
    # must use rgb_array so frames can be recorded
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # wrap with video recorder
    env = RecordVideo(env, "recordings")

    for episode in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        total_steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_steps += 1

        print(f"Episode {episode+1} done in {total_steps} steps, reward = {total_reward:.2f}")

    env.close()
