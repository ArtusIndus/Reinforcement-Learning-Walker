import pygame
from miniwalker_env import MiniWalkerEnv
import numpy as np

env = MiniWalkerEnv()

obs, _ = env.reset()

running = True

while running:

    action = [0.0, 0.0, 0.0, 0.0]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Linke Hüfte
    if keys[pygame.K_q]:
        action[0] = -1.0
    if keys[pygame.K_a]:
        action[0] = 1.0

    # Linkes Knie
    if keys[pygame.K_w]:
        action[1] = -1.0
    if keys[pygame.K_s]:
        action[1] = 1.0

    # Rechte Hüfte
    if keys[pygame.K_e]:
        action[2] = -1.0
    if keys[pygame.K_d]:
        action[2] = 1.0

    # Rechtes Knie
    if keys[pygame.K_r]:
        action[3] = -1.0
    if keys[pygame.K_f]:
        action[3] = 1.0

    action = np.array(action, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

    env.render()

pygame.quit()