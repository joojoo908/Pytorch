# --- ModelTest.py ---
# 시각화/평가 유틸: Gymnasium ENV + (학습된) Actor
# - 미로 격자 A*와 전역(128x128 등) A* 모두 지원
# - A* 경로가 None이어도 안전하게 처리 (이전 경로 유지 가정)
# - pygame 설치 없어도 headless로 동작

import sys
import time
import math
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
        # p: [x, y]
        x, y = float(p[0]), float(p[1])
        sx = int((x + map_range) * scale)
        sy = int((map_range - y) * scale)  # y-축 반전
        return sx, sy

    def rect_world_to_screen(center, half):
        """
        center: [cx, cy], half: [hx, hy]  (월드 단위)
        pygame.Rect에 넣을 top-left(x,y)와 width,height(픽셀) 반환
        """
        cx, cy = float(center[0]), float(center[1])
        hx, hy = float(half[0]),   float(half[1])
        minx = cx - hx
        miny = cy - hy
        w = 2.0 * hx
        h = 2.0 * hy

        sx = int((minx + map_range) * scale)
        # top-left y = (map_range - top_y)*scale, top_y = miny + h
        sy = int((map_range - (miny + h)) * scale)

        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        return pygame.Rect(sx, sy, sw, sh)

    return W, H, world_to_screen, rect_world_to_screen


# -------------------------------
# A* 경로(world 좌표 리스트) 추출
# -------------------------------
def astar_points_world(env):
    """
    env._astar_path 가 있을 때, 각 (r,c)을 월드좌표로 변환하여 리스트로 반환.
    - 전역 A*: _astar_origin + c*size, r*size
    - 미로 A*: _maze_origin  + c*size, r*size
    - 헬퍼 메서드가 있으면 우선 사용: _astar_cell_center_world, _cell_center_world
    """
    pts = []
    path = getattr(env, "_astar_path", None)
    if not path:
        return pts

    # 1) 전역 A* 헬퍼 메서드
    if hasattr(env, "_astar_cell_center_world") and callable(getattr(env, "_astar_cell_center_world")):
        for (r, c) in path:
            wp = env._astar_cell_center_world(r, c)
            pts.append((float(wp[0]), float(wp[1])))
        return pts

    # 2) 미로 A* 헬퍼 메서드
    if hasattr(env, "_cell_center_world") and callable(getattr(env, "_cell_center_world")):
        for (r, c) in path:
            wp = env._cell_center_world(r, c)
            pts.append((float(wp[0]), float(wp[1])))
        return pts

    # 3) 전역 A* 원시 파라미터
    if hasattr(env, "_astar_origin") and hasattr(env, "_astar_cell_size"):
        origin = np.array(env._astar_origin, dtype=np.float32)
        cell   = float(env._astar_cell_size)
        for (r, c) in path:
            wp = origin + np.array([c * cell, r * cell], dtype=np.float32)
            pts.append((float(wp[0]), float(wp[1])))
        return pts

    # 4) 미로 A* 원시 파라미터
    if hasattr(env, "_maze_origin") and hasattr(env, "_maze_cell_size"):
        origin = np.array(env._maze_origin, dtype=np.float32)
        cell   = float(env._maze_cell_size)
        for (r, c) in path:
            wp = origin + np.array([c * cell, r * cell], dtype=np.float32)
            pts.append((float(wp[0]), float(wp[1])))
        return pts

    # 실패 시 빈 리스트
    return pts


# -------------------------------
# 평가/시각화 루프
# -------------------------------
def evaluate_once(env,
                  actor: nn.Module,
                  max_steps: int = None,
                  scale: int = 20,
                  wait: int = 10,
                  visualize: bool = True,
                  auto_quit: bool = True):
    """
    - env: Gymnasium 호환 ENV
    - actor: nn.Module, 입력 shape=[1, obs_dim] -> 출력 shape=[1, action_dim], 범위 [-1,1]
    - max_steps: None이면 env.max_steps 또는 300
    - scale: 화면 배율(픽셀/월드단위)
    - wait: 종료 전 대기 프레임(시각화만)
    - visualize: False면 headless로 평가
    - auto_quit: True면 에피소드 끝나면 창 자동 종료
    """
    assert hasattr(env, "map_range"), "env.map_range 가 필요합니다."

    obs, info = env.reset()
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    # 안전 체크
    if isinstance(actor, nn.Module):
        actor.eval()

    if max_steps is None:
        max_steps = getattr(env, "max_steps", 300)

    # pygame 준비
    if visualize and HAS_PYGAME:
        pygame.init()
        W, H, world_to_screen, rect_world_to_screen = make_world_to_screen(env.map_range, scale)
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("ModelTest - A* Path Visualization")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 16)
    else:
        world_to_screen = lambda p: (0, 0)
        rect_world_to_screen = None
        screen = None
        clock = None
        font = None

    ep_ret = 0.0
    done = False

    for step in range(max_steps):
        # 이벤트 처리(ESC 종료)
        if visualize and HAS_PYGAME:
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
            # 콜러블 함수 지원
            a = np.asarray(actor(obs), dtype=np.float32)

        # 스텝
        obs, reward, terminated, truncated, info = env.step(a)
        ep_ret += float(reward)
        done = done or terminated or truncated

        # 시각화
        if visualize and HAS_PYGAME:
            screen.fill((18, 18, 18))

            # 벽(사각형들)
            wall_centers = getattr(env, "_wall_centers", None)
            wall_halves  = getattr(env, "_wall_halves",  None)
            if (wall_centers is not None) and (wall_halves is not None) and (rect_world_to_screen is not None):
                for i in range(wall_centers.shape[0]):
                    rect = rect_world_to_screen(wall_centers[i], wall_halves[i])
                    pygame.draw.rect(screen, (70, 70, 70), rect)

            # A* 경로 (연결선)
            pts_world = astar_points_world(env)
            if len(pts_world) >= 2:
                pts_screen = [world_to_screen(p) for p in pts_world]
                pygame.draw.lines(screen, (120, 220, 120), False, pts_screen, 2)

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

    # 에피소드 종료 후 대기
    if visualize and HAS_PYGAME and wait > 0:
        t0 = time.time()
        while time.time() - t0 < wait / 30.0:  # 대충 프레임 환산
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    t0 = 1e9
                    break
            pygame.time.delay(10)

    if visualize and HAS_PYGAME:
        if auto_quit:
            pygame.quit()

    return ep_ret


# -------------------------------
# 간단 실행 테스트
# -------------------------------
if __name__ == "__main__":
    # ENV와 연결 테스트용 (실제 사용에선 외부에서 env/actor 주입)
    try:
        import ENV

        env = ENV.Vector2DEnv(
            maze_cells=(15, 15),
            step_size=0.1,
            on_collision="deflect",
            R_SUCCESS=500.0,
            # A* 셀 길이 보상(새로 추가된 옵션)
            astar_shaping_scale=2.0,
            astar_shaping_clip=5.0,
            astar_grid=(256, 256),  # 전역 A* 격자 해상도
            astar_replan_steps=1  # A* 재계획 주기(스텝)
        )
    except Exception as e:
        print("[WARN] ENV 로드 실패 또는 생성 실패:", e)
        sys.exit(0)

    actor = DummyActor(env.observation_space.shape[0], env.action_space.shape[0]).eval()
    ret = evaluate_once(env, actor, scale=22, wait=30, visualize=HAS_PYGAME, auto_quit=True)
    print(f"Episode return: {ret:.3f}")
