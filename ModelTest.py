# --- ModelTest.py (deterministic eval, actor 전용 + 장애물 + 최단경로 + 남은 스텝/거리 표시) ---

import sys
import time
import os
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

import torch
import torch.nn as nn


# ============================================================
# Utils
# ============================================================

def make_world_to_screen(map_range, scale):
    W = int(2 * map_range * scale)
    H = int(2 * map_range * scale)

    def world_to_screen(p):
        x, y = float(p[0]), float(p[1])
        sx = int((x + map_range) * scale)
        sy = int((map_range - y) * scale)
        return sx, sy

    def rect_world_to_screen(center, half):
        import pygame
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


def get_geodesic_distance(env):
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


def recover_descent_path_world(env, max_len=500):
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


# ============================================================
# Deterministic action helper
# ============================================================

@torch.no_grad()
def policy_act(actor, obs_np):
    # 입력을 actor와 같은 device로 이동
    x = torch.from_numpy(obs_np).float().unsqueeze(0).to(next(actor.parameters()).device)
    out = actor.forward(x)
    if isinstance(out, (tuple, list)):
        mean = out[0]
        a = torch.tanh(mean)
    else:
        a = torch.clamp(out, -1.0, 1.0)
    return a.squeeze(0).cpu().numpy()


# ============================================================
# Evaluation loop
# ============================================================

def evaluate_once(env, actor, max_steps=None, scale=20,
                  screen_bundle=None, visualize=True):
    import pygame

    obs, info = env.reset()
    actor.eval()

    if max_steps is None:
        max_steps = getattr(env, "max_steps", 300)

    screen = clock = font = None
    world_to_screen = lambda p: (0,0)
    rect_world_to_screen = None

    if visualize and HAS_PYGAME:
        if screen_bundle is None:
            pygame.init()
            W,H,world_to_screen,rect_world_to_screen = make_world_to_screen(env.map_range, scale)
            screen = pygame.display.set_mode((W,H))
            pygame.display.set_caption("ModelTest - SAC Actor")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("consolas",16)
            screen_bundle=(screen,clock,font,world_to_screen,rect_world_to_screen)
        else:
            screen,clock,font,world_to_screen,rect_world_to_screen = screen_bundle

    ep_ret=0.0
    done=False

    for step in range(max_steps):
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type==pygame.QUIT: done=True
                elif event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE: done=True

        a = policy_act(actor, obs)
        obs,reward,terminated,truncated,info = env.step(a)
        ep_ret+=float(reward)
        done=done or terminated or truncated

        if visualize and HAS_PYGAME and screen is not None:
            screen.fill((18,18,18))

            # === 장애물 그리기 ===
            wall_centers = getattr(env, "_wall_centers", None)
            wall_halves  = getattr(env, "_wall_halves", None)
            if (wall_centers is not None) and (wall_halves is not None) and (rect_world_to_screen is not None):
                for i in range(wall_centers.shape[0]):
                    rect = rect_world_to_screen(wall_centers[i], wall_halves[i])
                    pygame.draw.rect(screen, (70, 70, 70), rect)

            # === 최단거리 경로 그리기 ===
            pts_world = recover_descent_path_world(env, max_len=500)
            if len(pts_world) >= 2:
                pts_screen = [world_to_screen(p) for p in pts_world]
                pygame.draw.lines(screen, (240, 180, 60), False, pts_screen, 2)

            # === 목표 & 에이전트 ===
            ag=np.array(env.agent_pos); gl=np.array(env.goal_pos)
            pygame.draw.circle(screen,(230,90,90),world_to_screen(gl),5)
            pygame.draw.circle(screen,(80,180,250),world_to_screen(ag),4)

            # === 남은 스텝 & 거리 표시 ===
            remaining = max_steps - step
            dist_to_goal = float(np.linalg.norm(env.goal_pos - env.agent_pos))
            info_lines = [
                f"Step: {step}/{max_steps} (남은 {remaining})",
                f"Dist to Goal: {dist_to_goal:.3f}"
            ]
            y = 5
            for line in info_lines:
                surf = font.render(line, True, (220,220,220))
                screen.blit(surf, (5, y))
                y += 18

            pygame.display.flip(); clock.tick(60)

        if done: break

    print(f"[Eval] ep_ret={ep_ret:.3f}")
    return ep_ret, screen_bundle


def run_multiple_evaluations(env, actor, episodes=5,
                             max_steps=None, scale=20,
                             visualize=True, visualize_every=1,
                             wait=20, auto_quit=True):
    returns=[]; screen_bundle=None
    for ep in range(episodes):
        vis = visualize and ((ep % visualize_every)==0)
        ret,screen_bundle = evaluate_once(env,actor,
            max_steps=max_steps,scale=scale,
            screen_bundle=screen_bundle if vis else None,
            visualize=vis)
        returns.append(ret)
        print(f"[Episode {ep+1}/{episodes}] return={ret:.3f}")
    if visualize and HAS_PYGAME and screen_bundle and auto_quit:
        import pygame; pygame.quit()
    return returns


# ============================================================
# Main
# ============================================================

if __name__=="__main__":
    import ENV
    from Model import GaussianPolicy, device

    actor_path="sac_actor.pth"
    env=ENV.Vector2DEnv(seed=1)

    if not os.path.exists(actor_path):
        print(f"[WARN] {actor_path} 없음. 먼저 Test.py로 학습 후 저장하세요.")
        sys.exit(0)

    # Actor만 로드
    actor=GaussianPolicy(env.observation_space.shape[0],
                         env.action_space.shape[0]).to(device)
    obj=torch.load(actor_path,map_location=device)
    actor.load_state_dict(obj)
    actor.eval()

    returns=run_multiple_evaluations(env,actor,
                                     episodes=3,scale=22,
                                     visualize=HAS_PYGAME,
                                     visualize_every=1,
                                     wait=20,auto_quit=True)
