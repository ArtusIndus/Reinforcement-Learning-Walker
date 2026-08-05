from stable_baselines3 import PPO
from miniwalker_env import MiniWalkerEnv

env = MiniWalkerEnv()

model = PPO.load("C:/Users/dick-/Projects/RLQuadruped/miniwalker/miniwalker.zip")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    if terminated or truncated:
        obs, _ = env.reset()

pygame.quit()