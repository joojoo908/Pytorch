# --- ModelTest.py ---
# 시각화/평가 유틸: Gymnasium ENV + (학습된) Actor
# - 지오데식(근사 최단거리) HUD 표시
# - from_start 모드: start_d / progress / inc HUD 추가
# - delta 모드: geo_d / Δd HUD
# - (G) 키: 지오데식 거리맵 히트맵 토글
# - (P) 키: 지오데식 하강 경로(에이전트→목표) 토글
# - pygame 없어도 headless로 동작 (콘솔 요약 출력)

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
        sy = int((map_range - (miny + h)) * scale)
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        return pygame.Rect(sx, sy, sw, sh)

    return W, H, world_to_screen, rect_world_to_screen


# -------------------------------
# 지오데식 거리/경로 유틸 (ENV의 "지오데식 전용 격자" 사용)
# -------------------------------
def get_geodesic_distance(env):
    """현재 에이전트 위치의 지오데식 거리를 반환. 없으면 None."""
    try:
        geo = getattr(env, "_geo_map", None)
        if geo is None or not np.isfinite(geo).any():
            return None
        if hasattr(env, "_pos_to_geo_rc"):
            r, c = env._pos_to_geo_rc(env.agent_pos)
        else:
            return None
        d = float(geo[r, c])
        return d if np.isfinite(d) else None
    except Exception:
        return None


def get_geodesic_init(env):
    """에피소드 시작 시 지오데식 거리(d_init) 반환(있으면)."""
    val = getattr(env, "_geo_init", None)
    return float(val) if val is not None else None


def get_geodesic_progress_given(env):
    """from_start 모드에서 지금까지 누적 지급된 진행량(best progress) 반환(있으면)."""
    val = getattr(env, "_geo_progress_given", None)
    return float(val) if val is not None else None


def recover_descent_path_world(env, max_len=500):
    """
    지오데식 거리맵을 따라 '내리막'으로 이동하며
    에이전트→목표 하강 경로(월드 좌표 리스트) 복원.
    (ENV의 128x128 전용 격자 메타 사용)
    """
    geo = getattr(env, "_geo_map", None)
    if geo is None:
        return []

    origin = getattr(env, "_geo_origin", None)
    cellsz = getattr(env, "_geo_cell_size", None)
    goal_rc = getattr(env, "_geo_goal_rc", None)
    if origin is None or cellsz is None:
        return []

    if hasattr(env, "_pos_to_geo_rc"):
        r, c = env._pos_to_geo_rc(env.agent_pos)
    else:
        return []

    rows, cols = geo.shape
    cw, ch = float(cellsz[0]), float(cellsz[1])

    def rc_to_world(rr, cc):
        return origin + np.array([cc * cw, rr * ch], dtype=np.float32)

    cur = (int(r), int(c))
    if not (0 <= cur[0] < rows and 0 <= cur[1] < cols):
        return []
    if not np.isfinite(geo[cur[0], cur[1]]):
        return []

    pts = [rc_to_world(cur[0], cur[1])]
    last_val = float(geo[cur[0], cur[1]])
    N8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    for _ in range(max_len):
        if goal_rc is not None and cur == goal_rc:
            break
        best = (last_val, None)
        rr, cc = cur
        for dr, dc in N8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = float(geo[nr, nc])
                if np.isfinite(v) and v + 1e-6 < best[0]:
                    best = (v, (nr, nc))
        if best[1] is None:
            break
        cur = best[1]
        last_val = best[0]
        pts.append(rc_to_world(cur[0], cur[1]))

    return pts


def draw_geodesic_heatmap(screen, env, rect_world_to_screen):
    geo = getattr(env, "_geo_map", None)
    if geo is None:
        return
    origin = getattr(env, "_geo_origin", None)
    cellsz = getattr(env, "_geo_cell_size", None)
    if origin is None or cellsz is None:
        return

    cw, ch = float(cellsz[0]), float(cellsz[1])
    half = np.array([cw * 0.5, ch * 0.5], dtype=np.float32)

    finite_vals = geo[np.isfinite(geo)]
    if finite_vals.size == 0:
        return
    gmin, gmax = float(finite_vals.min()), float(finite_vals.max())
    rng = max(1e-6, gmax - gmin)

    rows, cols = geo.shape
    for r in range(rows):
        for c in range(cols):
            v = float(geo[r, c])
            if not np.isfinite(v):
                color = (90, 40, 110)  # 도달 불가
            else:
                t = (v - gmin) / rng
                g = int(40 + 180 * (1.0 - t))  # 가까울수록 어둡게
                color = (g, g, g)
            ctr = origin + np.array([c * cw, r * ch], dtype=np.float32)
            rect = rect_world_to_screen(ctr, half)
            pygame.draw.rect(screen, color, rect)


# -------------------------------
# 평가/시각화 루프
# -------------------------------
def evaluate_once(env,
                  actor: nn.Module,
                  max_steps: int = None,
                  scale: int = 20,
                  screen_bundle=None,   # (screen, clock, font, world_to_screen, rect_world_to_screen)
                  visualize: bool = True,
                  geo_heatmap: bool = False,
                  geo_path: bool = True):
    """
    - env: Gymnasium 호환 ENV
    - actor: nn.Module, 입력 shape=[1, obs_dim] -> 출력 shape=[1, action_dim], 범위 [-1,1]
    - max_steps: None이면 env.max_steps 또는 300
    - scale: 화면 배율(픽셀/월드단위)
    - screen_bundle: 외부에서 만든 pygame 화면 리소스를 재사용
    - visualize: False면 headless로 평가
    - geo_heatmap: 지오데식 거리맵 히트맵 표시
    - geo_path: 지오데식 하강 경로(에이전트→목표) 표시
    """
    obs, info = env.reset()

    if isinstance(actor, nn.Module):
        actor.eval()

    if max_steps is None:
        max_steps = getattr(env, "max_steps", 300)

    screen = clock = font = None
    world_to_screen = lambda p: (0, 0)
    rect_world_to_screen = None

    if visualize and HAS_PYGAME:
        if screen_bundle is None:
            pygame.init()
            W, H, world_to_screen, rect_world_to_screen = make_world_to_screen(env.map_range, scale)
            screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("ModelTest - Geodesic Visualization")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("consolas", 16)
            screen_bundle = (screen, clock, font, world_to_screen, rect_world_to_screen)
        else:
            screen, clock, font, world_to_screen, rect_world_to_screen = screen_bundle

    ep_ret = 0.0
    done = False

    geo_mode = getattr(env, "geo_mode", "delta")
    geo_init = get_geodesic_init(env)
    last_geo = get_geodesic_distance(env)

    geo_heatmap_on = bool(geo_heatmap)
    geo_path_on = bool(geo_path)

    for step in range(max_steps):
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_g:
                        geo_heatmap_on = not geo_heatmap_on
                    elif event.key == pygame.K_p:
                        geo_path_on = not geo_path_on

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

        # 보상 항목들(있을 때만)
        terms = info.get("reward_terms", {}) if isinstance(info, dict) else {}
        geo_from_start = terms.get("geo_from_start", None)
        geo_increment  = terms.get("geo_increment", None)
        geo_delta      = terms.get("geo_delta", None)

        # 지오데식 거리/변화량
        geo_now = info.get("geo_dist", None)
        if geo_now is None:
            geo_now = get_geodesic_distance(env)
        step_delta = None
        if last_geo is not None and geo_now is not None:
            step_delta = last_geo - geo_now
        last_geo = geo_now

        # 시각화
        if visualize and HAS_PYGAME and screen is not None:
            screen.fill((18, 18, 18))

            # 지오데식 히트맵
            if geo_heatmap_on and rect_world_to_screen is not None:
                draw_geodesic_heatmap(screen, env, rect_world_to_screen)

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

            # 지오데식 하강 경로
            if geo_path_on:
                pts_world = recover_descent_path_world(env, max_len=500)
                if len(pts_world) >= 2:
                    pts_screen = [world_to_screen(p) for p in pts_world]
                    pygame.draw.lines(screen, (240, 180, 60), False, pts_screen, 2)

            # HUD
            lines = []
            lines.append(f"step:{step}  R:{reward:.3f}  ep_ret:{ep_ret:.3f}")

            if geo_mode == "from_start":
                prog_best = get_geodesic_progress_given(env)
                # 1줄: 시작거리 & 현재거리
                g0 = f"{geo_init:.3f}" if geo_init is not None else "-"
                gn = f"{geo_now:.3f}" if geo_now is not None else "-"
                lines.append(f"mode:from_start  start_d:{g0}  geo_d:{gn}")
                # 2줄: 누적진행/이번증분 (ENV가 제공하면 그 값, 아니면 계산값 보조)
                if geo_from_start is not None or geo_increment is not None or prog_best is not None:
                    fs = f"{geo_from_start:.3f}" if geo_from_start is not None else "-"
                    inc = f"{geo_increment:+.3f}" if geo_increment is not None else "-"
                    pb = f"{prog_best:.3f}" if prog_best is not None else "-"
                    lines.append(f"progress:{fs}  inc:{inc}  best:{pb}")
                elif (geo_init is not None) and (geo_now is not None):
                    fs = max(0.0, geo_init - geo_now)
                    lines.append(f"progress:{fs:.3f}  inc:-  best:-")
            else:
                # delta 모드
                gn = f"{geo_now:.3f}" if geo_now is not None else "-"
                sd = f"{step_delta:+.3f}" if step_delta is not None else "-"
                # ENV가 geo_delta 제공하면 그것도 보여주기
                gd = f"{geo_delta:+.3f}" if geo_delta is not None else "-"
                lines.append(f"mode:delta  geo_d:{gn}  Δd(step):{sd}  Δd(env):{gd}")

            # 토글 안내
            lines.append("[G]heatmap " + ("ON" if geo_heatmap_on else "OFF") +
                         "   [P]path " + ("ON" if geo_path_on else "OFF"))

            # 그리기
            if font is not None:
                y = 8
                for i, t in enumerate(lines):
                    color = (220, 220, 220) if i < len(lines)-1 else (160, 200, 220)
                    surf = font.render(t, True, color)
                    screen.blit(surf, (8, y))
                    y += 20

            pygame.display.flip()
            clock.tick(60)

        if done:
            break

    if not visualize or not HAS_PYGAME:
        if geo_mode == "from_start":
            prog_best = get_geodesic_progress_given(env)
            print(f"[Eval] ep_ret={ep_ret:.3f}, geo_init={geo_init}, best_progress={prog_best}, last_geo={last_geo}")
        else:
            print(f"[Eval] ep_ret={ep_ret:.3f}, last_geo={last_geo}")

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
                             visualize_every: int = 1,
                             wait: int = 20,
                             auto_quit: bool = True,
                             geo_heatmap: bool = False,
                             geo_path: bool = True):
    returns = []
    screen_bundle = None

    for ep in range(episodes):
        vis = visualize and ((ep % visualize_every) == 0)
        ret, screen_bundle = evaluate_once(
            env, actor,
            max_steps=max_steps,
            scale=scale,
            screen_bundle=screen_bundle if vis else None,
            visualize=vis,
            geo_heatmap=geo_heatmap,
            geo_path=geo_path
        )
        returns.append(ret)
        print(f"[Episode {ep+1}/{episodes}] return = {ret:.3f}")

        if vis and HAS_PYGAME and wait > 0 and screen_bundle is not None:
            screen, clock, font, world_to_screen, rect_world_to_screen = screen_bundle
            t0 = time.time()
            while time.time() - t0 < wait / 60.0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        t0 = 1e9
                        break
                pygame.time.delay(10)

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
    try:
        import ENV
        # 보기만 원하면 geodesic_coef=0.0; 학습 확인은 0.2~0.5 추천
        env = ENV.Vector2DEnv(
            geodesic_shaping=True,
            geodesic_progress_mode="delta",
            geodesic_coef=0.3,
            step_size=0.25,

            # ★ 근접 패널티 on (기본값 그대로여도 됨)
            proximity_penalty=True,
            proximity_threshold=0.20,
            proximity_coef=0.3,  # 너무 크면 목표 보상에 비해 학습이 소심해질 수 있음
            proximity_clip=0.2,

            # ★ 충돌 종료 off (기본값 False)
            collision_terminate=True,
            seed=28
        )

    except Exception as e:
        print("[WARN] ENV 로드 실패 또는 생성 실패:", e)
        sys.exit(0)

    actor = DummyActor(env.observation_space.shape[0], env.action_space.shape[0]).eval()

    returns = run_multiple_evaluations(
        env, actor,
        episodes=3,
        scale=22,
        visualize=HAS_PYGAME,
        visualize_every=1,
        wait=20,
        auto_quit=True,
        geo_heatmap=True,
        geo_path=True
    )
