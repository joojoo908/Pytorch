# --- ModelTest.py ---
# 시각화/평가 유틸: Gymnasium ENV + (학습된) Actor
# - A* 관련 코드 제거 (ENV가 A*를 사용하지 않음)
# - pygame 설치 없어도 headless로 동작
# - 여러 에피소드를 연속 실행 가능

import sys
import time
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 간단한 더미 액터 (학습 모델 없을 때 테스트용)
# -------------------------------
class DummyActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu  = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # [-1,1] 범위
        return torch.tanh(self.mu(x))


# -------------------------------
# 월드 <-> 스크린 변환 유틸
# -------------------------------
def make_world_to_screen(map_range, scale):
    """
    월드 좌표계: x,y ∈ [-map_range, +map_range]
    스크린 좌표계: (0,0) 좌상단, y 아래로 증가 (pygame 규칙)
    """
    W = int(2 * map_range * scale)
    H = int(2 * map_range * scale)

    def world_to_screen(p):
        x, y = float(p[0]), float(p[1])
        sx = int((x + map_range) * scale)
        sy = int((map_range - y) * scale)  # y-축 반전
        return sx, sy

    def rect_world_to_screen(center, half):
        cx, cy = float(center[0]), float(center[1])
        hx, hy = float(half[0]),   float(half[1])
        minx = cx - hx
        miny = cy - hy
        w = 2.0 * hx
        h = 2.0 * hy

        sx = int((minx + map_range) * scale)
        sy = int((map_range - (miny + h)) * scale)  # top-left
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        return pygame.Rect(sx, sy, sw, sh)

    return W, H, world_to_screen, rect_world_to_screen


# -------------------------------
# 단일 에피소드 평가/시각화 (창 재사용 지원)
# -------------------------------
def evaluate_once(env,
                  actor: nn.Module,
                  max_steps: int = None,
                  scale: int = 20,
                  screen_bundle=None,   # (screen, clock, font, world_to_screen, rect_world_to_screen)
                  visualize: bool = True):
    """
    - env: Gymnasium 호환 ENV
    - actor: nn.Module, 입력 shape=[1, obs_dim] -> 출력 shape=[1, action_dim], 범위 [-1,1]
    - max_steps: None이면 env.max_steps 또는 300
    - scale: 화면 배율(픽셀/월드단위)
    - screen_bundle: 외부에서 만든 pygame 화면 리소스를 재사용
    - visualize: False면 headless로 평가
    """
    obs, info = env.reset()

    # 안전 체크
    if isinstance(actor, nn.Module):
        actor.eval()

    if max_steps is None:
        max_steps = getattr(env, "max_steps", 300)

    # pygame 준비/재사용
    screen = clock = font = None
    world_to_screen = lambda p: (0, 0)
    rect_world_to_screen = None

    if visualize and HAS_PYGAME:
        if screen_bundle is None:
            pygame.init()
            W, H, world_to_screen, rect_world_to_screen = make_world_to_screen(env.map_range, scale)
            screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("ModelTest - Visualization")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("consolas", 16)
            screen_bundle = (screen, clock, font, world_to_screen, rect_world_to_screen)
        else:
            screen, clock, font, world_to_screen, rect_world_to_screen = screen_bundle

    ep_ret = 0.0
    done = False

    for step in range(max_steps):
        # 이벤트 처리(ESC 종료)
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

        # 액터 추론
        if isinstance(actor, nn.Module):
            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0)
                a = actor(x).squeeze(0).cpu().numpy()
        else:
            a = np.asarray(actor(obs), dtype=np.float32)

        # 스텝
        obs, reward, terminated, truncated, info = env.step(a)
        ep_ret += float(reward)
        done = done or terminated or truncated

        # 시각화
        if visualize and HAS_PYGAME and screen is not None:
            screen.fill((18, 18, 18))

            # 벽(사각형들)
            wall_centers = getattr(env, "_wall_centers", None)
            wall_halves  = getattr(env, "_wall_halves",  None)
            if (wall_centers is not None) and (wall_halves is not None) and (rect_world_to_screen is not None):
                for i in range(wall_centers.shape[0]):
                    rect = rect_world_to_screen(wall_centers[i], wall_halves[i])
                    pygame.draw.rect(screen, (70, 70, 70), rect)

            # 목표/에이전트
            ag = np.array(getattr(env, "agent_pos"), dtype=np.float32)
            gl = np.array(getattr(env, "goal_pos"),  dtype=np.float32)
            ag_s = world_to_screen(ag)
            gl_s = world_to_screen(gl)

            pygame.draw.circle(screen, (230, 90, 90), gl_s, max(3, int(0.15 * scale)))
            pygame.draw.circle(screen, (80, 180, 250), ag_s, max(3, int(0.12 * scale)))

            # 텍스트 HUD
            txt = f"step:{step}  R:{reward:.3f}  ep_ret:{ep_ret:.3f}"
            if font is not None:
                surf = font.render(txt, True, (220, 220, 220))
                screen.blit(surf, (8, 8))

            pygame.display.flip()
            clock.tick(60)

        if done:
            break

    return ep_ret, screen_bundle


# -------------------------------
# 여러 에피소드 연속 평가 (창 1개 재사용)
# -------------------------------
def run_multiple_evaluations(env,
                             actor: nn.Module,
                             episodes: int = 5,
                             max_steps: int = None,
                             scale: int = 20,
                             visualize: bool = True,
                             visualize_every: int = 1,  # n 에피소드마다 1번 시각화
                             wait: int = 20,            # 에피소드 사이/끝 대기 프레임(시각화 모드에서만)
                             auto_quit: bool = True):
    """
    - visualize_every: 1이면 모든 에피소드 시각화, 2면 2개마다 1번만 시각화 등
    - wait: 한 에피소드 끝나고 다음 에피소드 시작 전 짧게 기다림 (시각화 보기 좋게)
    """
    returns = []
    screen_bundle = None

    for ep in range(episodes):
        vis = visualize and ((ep % visualize_every) == 0)
        ret, screen_bundle = evaluate_once(
            env, actor,
            max_steps=max_steps,
            scale=scale,
            screen_bundle=screen_bundle if vis else None,
            visualize=vis
        )
        returns.append(ret)
        print(f"[Episode {ep+1}/{episodes}] return = {ret:.3f}")

        # 에피소드 사이 짧은 대기 (시각화일 때만)
        if vis and HAS_PYGAME and wait > 0 and screen_bundle is not None:
            screen, clock, font, world_to_screen, rect_world_to_screen = screen_bundle
            t0 = time.time()
            while time.time() - t0 < wait / 60.0:  # 대충 프레임 환산
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        t0 = 1e9
                        break
                pygame.time.delay(10)

    # 종료 정리
    if visualize and HAS_PYGAME and screen_bundle is not None and auto_quit:
        pygame.quit()

    if len(returns) > 0:
        avg = sum(returns) / len(returns)
        print(f"[Summary] episodes={episodes}, avg_return={avg:.3f}, min={min(returns):.3f}, max={max(returns):.3f}")
    return returns


# -------------------------------
# 간단 실행 테스트
# -------------------------------
if __name__ == "__main__":
    # ENV와 연결 테스트용 (실제 사용에선 외부에서 env/actor 주입)
    try:
        import ENV
        # 맵은 고정, 시작/목표는 매 에피소드 랜덤
        env = ENV.Vector2DEnv(seed=42, fixed_maze=True, fixed_agent_goal=False)
    except Exception as e:
        print("[WARN] ENV 로드 실패 또는 생성 실패:", e)
        sys.exit(0)

    actor = DummyActor(env.observation_space.shape[0], env.action_space.shape[0]).eval()

    # 여러 번 실행: 모든 에피소드 시각화(visualize_every=1), 에피소드 사이 20프레임 대기
    returns = run_multiple_evaluations(
        env, actor,
        episodes=5,
        scale=22,
        visualize=HAS_PYGAME,
        visualize_every=1,
        wait=20,
        auto_quit=True
    )

    # headless로 빠르게만 보고 싶다면:
    # returns = run_multiple_evaluations(env, actor, episodes=20, visualize=False)

    # 단일 에피소드만 돌리고 싶다면:
    # ret, _ = evaluate_once(env, actor, scale=22, visualize=HAS_PYGAME)
