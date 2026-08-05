from pathlib import Path

from stable_baselines3 import PPO
from miniwalker_env import MiniWalkerEnv


MODEL_PATH = Path("miniwalker.zip")
TIMESTEPS = 1_000_000


env = MiniWalkerEnv()

if MODEL_PATH.exists():
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("Creating new model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

print(f"Training for {TIMESTEPS:,} timesteps...")

model.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=False
)

model.save(MODEL_PATH)

print("Training finished.")
print(f"Model saved to: {MODEL_PATH.resolve()}")