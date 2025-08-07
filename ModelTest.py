import pygame
import numpy as np
import torch

import Model
import ENV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_with_pygame(env, actor, scale=2.3, wait=10):
    pygame.init()
    size = (600, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("SAC Agent Navigation (Real-Time)")
    clock = pygame.time.Clock()

    actor.eval()
    running = True

    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    agent_path = [env.agent_pos.copy()]
    goal_pos = env.goal_pos.copy()

    def world_to_screen(pos):
        return int(size[0] / 2 + pos[0] * scale), int(size[1] / 2 - pos[1] * scale)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 모델로 행동 선택
        with torch.no_grad():
            action, _ = actor.sample(state_tensor)
        action = action.cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        agent_path.append(env.agent_pos.copy())
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 그리기
        screen.fill((255, 255, 255))

        # 경로 그리기
        if len(agent_path) > 1:
            for i in range(1, len(agent_path)):
                pygame.draw.line(screen, (0, 150, 255),
                                 world_to_screen(agent_path[i-1]),
                                 world_to_screen(agent_path[i]), 2)

        # 목표 위치
        pygame.draw.circle(screen, (255, 0, 0), world_to_screen(goal_pos), 8)

        # 시작 위치
        pygame.draw.circle(screen, (0, 255, 0), world_to_screen(agent_path[0]), 6)

        # 현재 위치
        pygame.draw.circle(screen, (0, 0, 255), world_to_screen(env.agent_pos), 6)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS
        pygame.time.delay(wait)

        if done:
            print("✔ 목표 도달 또는 종료 조건 만족")
            pygame.time.delay(1000)
            running = False

    pygame.quit()

env = ENV.Vector2DEnv(map_range=12.8, step_size=0.1)
actor = Model.GaussianPolicy(state_dim=4, action_dim=2).to(device)
actor.load_state_dict(torch.load("sac_actor.pth")) #모델 불러오기
actor.eval()

evaluate_with_pygame(env, actor,scale=20)