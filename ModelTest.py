# --- ModelTest.py (업데이트) ---

import pygame
import numpy as np
import torch
import Model
import ENV
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 유틸: 배우 입력 차원을 체크포인트에서 추론 =====
def infer_actor_input_dim_from_state_dict(sd: dict) -> int | None:
    # GaussianPolicy의 첫 Linear는 fc[0] 이고 키는 'fc.0.weight'일 가능성이 높음
    # 모듈명이 바뀌었을 경우를 대비해 모든 Linear weight에서 in_features 후보를 찾음
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2:  # [out_features, in_features]
            # 첫 레이어 후보: 보통 가장 작은 in_features(입력 차원)이거나 이름상 fc.0.weight
            if "fc.0.weight" in k:
                return v.shape[1]
    # fc.0.weight가 없다면 전체 중 가장 작은 in_features를 입력차원으로 추정
    candidates = [v.shape[1] for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    return min(candidates) if candidates else None


# ===== 유틸: 배우 모듈에서 첫 Linear의 in_features 읽기 =====
def actor_input_dim_from_module(actor: torch.nn.Module) -> int:
    for m in actor.modules():
        if isinstance(m, torch.nn.Linear):
            return m.in_features
    raise RuntimeError("Actor has no Linear layer to infer input dim.")


# ===== 유틸: 상태를 배우가 기대하는 입력 차원에 맞게 조정 =====
def adapt_state_for_actor(state_np: np.ndarray, expected_dim: int) -> np.ndarray:
    cur = state_np.shape[0]
    if cur == expected_dim:
        return state_np
    elif cur > expected_dim:
        # 앞쪽 피처를 우선 사용 (agent, goal 먼저이므로 의미 보존)
        return state_np[:expected_dim]
    else:
        # 모자라면 0 패딩
        pad = np.zeros(expected_dim - cur, dtype=state_np.dtype)
        return np.concatenate([state_np, pad], axis=0)


def evaluate(env, actor, scale=2.3, wait=10, auto_quit=True):
    pygame.init()
    size = (600, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("SAC Agent Navigation (Real-Time)")
    clock = pygame.time.Clock()

    actor.eval()
    running = True

    state, _ = env.reset()
    # 배우가 실제 기대하는 입력 차원
    expected_dim = actor_input_dim_from_module(actor)
    state = adapt_state_for_actor(state, expected_dim)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    agent_path = [env.agent_pos.copy()]
    goal_pos = env.goal_pos.copy()
    # 장애물은 고정이라면 reset 시점 복사, (움직이는 장애물로 바꾸면 env.obstacles를 매 프레임 참조)
    obstacles = env.obstacles.copy() if hasattr(env, "obstacles") and env.obstacles is not None else np.zeros((0, 2), dtype=np.float32)

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

        # 상태 업데이트(배우 입력 차원에 맞춤)
        state = adapt_state_for_actor(state, expected_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # --- 시각화 ---
        screen.fill((255, 255, 255))

        # 이동 경로
        if len(agent_path) > 1:
            for i in range(1, len(agent_path)):
                pygame.draw.line(
                    screen, (0, 150, 255),
                    world_to_screen(agent_path[i - 1]),
                    world_to_screen(agent_path[i]), 2
                )

        # 목표/시작/현재
        pygame.draw.circle(screen, (255, 0, 0), world_to_screen(goal_pos), 8)          # 목표
        pygame.draw.circle(screen, (0, 255, 0), world_to_screen(agent_path[0]), 6)     # 시작
        pygame.draw.circle(screen, (0, 0, 255), world_to_screen(env.agent_pos), 6)     # 현재

        # 장애물(회색 원)
        if obstacles is not None and len(obstacles) > 0 and hasattr(env, "obstacle_radius"):
            r_px = max(1, int(env.obstacle_radius * scale))
            for obs in obstacles:
                pygame.draw.circle(screen, (128, 128, 128), world_to_screen(obs), r_px)

        pygame.display.flip()
        clock.tick(60)
        pygame.time.delay(wait)

    dist = np.linalg.norm(env.goal_pos - env.agent_pos)
    success = dist < getattr(env, "threshold", 0.25)

    if success:
        print(f"✔ 목표 도달 성공! 걸린 스텝 수: {env.steps} / {env.max_steps}")
    else:
        print(f"✘ 목표 도달 실패. 최대 스텝 도달 ({env.max_steps}스텝 사용)")

    if auto_quit:
        pygame.time.delay(1000)
        pygame.quit()

    return success, env.steps


def run_multiple_evaluations(model_path="sac_actor.pth", episodes=10):
    # 장애물 포함 환경으로 생성 (원하는 개수/반경 조절)
    env = ENV.Vector2DEnv(map_range=12.8, step_size=0.1, num_obstacles=5, obstacle_radius=0.5)
    # 맵 크기에 따라 자동 스케일
    scale = 600 / (env.map_range * 2)

    # 체크포인트에서 배우 입력 차원을 추론하여 그에 맞게 네트워크 구성
    sd = torch.load(model_path, map_location=device)
    exp_dim = infer_actor_input_dim_from_state_dict(sd)
    if exp_dim is None:
        # 추론 실패 시 환경 상태 차원으로 가정
        exp_dim = env.observation_space.shape[0]

    actor = Model.GaussianPolicy(state_dim=exp_dim, action_dim=env.action_space.shape[0]).to(device)

    # 가중치 로드: 구조가 정확히 맞으면 strict=True, 일부 불일치 대비해 False
    try:
        actor.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[warn] strict=True 로드 실패: {e}\n → strict=False로 재시도합니다.")
        actor.load_state_dict(sd, strict=False)

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
    run_multiple_evaluations(model_path=model_path, episodes=10)
