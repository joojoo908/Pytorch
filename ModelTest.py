import pygame
import numpy as np
import torch
import Model
import ENV
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(env, actor, scale=2.3, wait=10, auto_quit=True):
    pygame.init()
    size = (600, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("SAC Agent Navigation (Real-Time)")
    clock = pygame.time.Clock()

    actor.eval()
    running = True

    state, _ = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    agent_path = [env.agent_pos.copy()]
    goal_pos = env.goal_pos.copy()

    def world_to_screen(pos):
        return int(size[0] / 2 + pos[0] * scale), int(size[1] / 2 - pos[1] * scale)

    done = False
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            action, _ = actor.sample(state_tensor)
        action = action.cpu().numpy()[0]
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent_path.append(env.agent_pos.copy())
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 시각화
        screen.fill((255, 255, 255))

        if len(agent_path) > 1:
            for i in range(1, len(agent_path)):
                pygame.draw.line(screen, (0, 150, 255),
                                 world_to_screen(agent_path[i-1]),
                                 world_to_screen(agent_path[i]), 2)

        pygame.draw.circle(screen, (255, 0, 0), world_to_screen(goal_pos), 8)  # 목표
        pygame.draw.circle(screen, (0, 255, 0), world_to_screen(agent_path[0]), 6)  # 시작
        pygame.draw.circle(screen, (0, 0, 255), world_to_screen(env.agent_pos), 6)  # 현재

        pygame.display.flip()
        clock.tick(60)
        pygame.time.delay(wait)

    dist = np.linalg.norm(env.goal_pos - env.agent_pos)
    success = dist < env.threshold

    if success:
        print(f"✔ 목표 도달 성공! 걸린 스텝 수: {env.steps} / {env.max_steps}")
    else:
        print(f"✘ 목표 도달 실패. 최대 스텝 도달 ({env.max_steps}스텝 사용)")

    if auto_quit:
        pygame.time.delay(1000)
        pygame.quit()

    return success, env.steps


def run_multiple_evaluations(model_path="sac_actor.pth", episodes=10):
    env = ENV.Vector2DEnv(map_range=12.8, step_size=0.1)
    scale = 600 / (env.map_range * 2)  # 맵 크기에 따라 자동 스케일

    actor = Model.GaussianPolicy(state_dim=4, action_dim=2).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    success_count = 0
    total_steps = 0

    for i in range(episodes):
        print(f"\n🌟 에피소드 {i + 1} 시작")
        success, steps = evaluate(env, actor, scale=scale, wait=10, auto_quit=(i == episodes - 1))
        success_count += int(success)
        total_steps += steps

    print("\n📊 평가 요약:")
    print(f"- 총 에피소드 수: {episodes}")
    print(f"- 성공률: {success_count / episodes * 100:.2f}%")
    print(f"- 평균 스텝 수: {total_steps / episodes:.2f}")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "sac_actor.pth"
    run_multiple_evaluations(model_path=model_path, episodes=5)
