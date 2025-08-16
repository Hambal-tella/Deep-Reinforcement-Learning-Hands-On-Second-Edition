import gymnasium as gym
import time

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    try:
        for episode in range(1, 10):  # 5 episodes
            obs, info = env.reset()
            total_reward = 0.0
            total_steps = 0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_steps += 1
                done = terminated or truncated
                time.sleep(0.02)  # slow down so you can see it

            print(f"Episode {episode} done in {total_steps} steps, total reward {total_reward:.2f}")
    finally:
        env.close()
