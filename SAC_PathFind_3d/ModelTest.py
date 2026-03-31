import os
import sys
from pathlib import Path

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

import torch


def make_world_to_screen(bounds_min, bounds_max, scale):
    min_x, min_z = float(bounds_min[0]), float(bounds_min[1])
    max_x, max_z = float(bounds_max[0]), float(bounds_max[1])
    width = max(1, int((max_x - min_x) * scale))
    height = max(1, int((max_z - min_z) * scale))

    def world_to_screen(p):
        x, z = float(p[0]), float(p[1])
        sx = int((x - min_x) * scale)
        sy = int((max_z - z) * scale)
        return sx, sy

    return width, height, world_to_screen


def get_geodesic_distance(env):
    fn = getattr(env, "_geo_distance_robust", None)
    if callable(fn):
        try:
            d = fn(env.agent_pos, max_search=3)
            return float(d) if d is not None else None
        except Exception:
            return None
    return None


def recover_descent_path_world(env, max_len=512):
    geo = getattr(env, "_geo_map", None)
    if geo is None:
        return []
    pos_to_rc = getattr(env, "_pos_to_geo_rc", None)
    rc_to_world = getattr(env, "_grid_rc_to_world", None)
    goal_rc = getattr(env, "_geo_goal_rc", None)
    if not callable(pos_to_rc) or not callable(rc_to_world):
        return []

    rows, cols = geo.shape
    start_r, start_c = pos_to_rc(env.agent_pos)

    def find_valid_start(r, c, radius=3):
        if 0 <= r < rows and 0 <= c < cols and np.isfinite(geo[r, c]):
            return r, c
        for rad in range(1, radius + 1):
            r0 = max(0, r - rad)
            r1 = min(rows - 1, r + rad)
            c0 = max(0, c - rad)
            c1 = min(cols - 1, c + rad)
            best = None
            best_val = np.inf
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    v = float(geo[rr, cc])
                    if np.isfinite(v) and v < best_val:
                        best = (rr, cc)
                        best_val = v
            if best is not None:
                return best
        return None

    cur = find_valid_start(start_r, start_c, radius=3)
    if cur is None:
        return []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    pts = [rc_to_world(cur[0], cur[1])]
    cur_val = float(geo[cur[0], cur[1]])

    for _ in range(max_len):
        if goal_rc is not None and cur == goal_rc:
            break
        best = None
        best_val = cur_val
        for dr, dc in neighbors:
            rr = cur[0] + dr
            cc = cur[1] + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            v = float(geo[rr, cc])
            if np.isfinite(v) and v + 1e-6 < best_val:
                best = (rr, cc)
                best_val = v
        if best is None:
            break
        cur = best
        cur_val = best_val
        pts.append(rc_to_world(cur[0], cur[1]))
    return pts


@torch.no_grad()
def policy_act(actor, obs_np):
    device = next(actor.parameters()).device
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

    if hasattr(actor, "act_deterministic"):
        a = actor.act_deterministic(x)
    else:
        out = actor(x)
        if isinstance(out, (tuple, list)):
            a = torch.tanh(out[0])
        else:
            a = torch.clamp(out, -1.0, 1.0)

    return a.squeeze(0).cpu().numpy()


def draw_navmesh_overlay(screen, env, world_to_screen):
    walkable = getattr(env, "_walkable", None)
    rc_to_world = getattr(env, "_grid_rc_to_world", None)
    cell_size = getattr(env, "_grid_cell_size", None)
    if walkable is None or not callable(rc_to_world) or cell_size is None:
        return

    rows, cols = walkable.shape
    if rows * cols > 120000:
        return

    step = max(1, int(round(float(cell_size) * 0.5)))
    color = (38, 48, 58)
    for r in range(rows):
        for c in range(cols):
            if walkable[r, c] == 0:
                continue
            world = rc_to_world(r, c)
            sx, sy = world_to_screen(world)
            pygame.draw.circle(screen, color, (sx, sy), step)


def evaluate_once(env, actor, max_steps=None, scale=0.03, screen_bundle=None, visualize=True, save_csv_path=None):
    obs, info = env.reset()
    actor.eval()

    start_pos = np.array(env.agent_pos, dtype=np.float32).copy()
    goal_pos = np.array(env.goal_pos, dtype=np.float32).copy()
    max_steps = int(max_steps or getattr(env, "max_steps", 300))

    traj = [start_pos.copy()]
    screen = clock = font = None
    world_to_screen = None

    if visualize and HAS_PYGAME:
        if screen_bundle is None:
            pygame.init()
            width, height, world_to_screen = make_world_to_screen(env.bounds_min, env.bounds_max, scale)
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("ModelTest - Majestro NavMesh")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("consolas", 16)
            screen_bundle = (screen, clock, font, world_to_screen)
        else:
            screen, clock, font, world_to_screen = screen_bundle

    ep_ret = 0.0
    final_info = {}
    terminated = truncated = False

    for step in range(max_steps):
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    terminated = True

        action = policy_act(actor, obs)
        obs, reward, terminated, truncated, final_info = env.step(action)
        ep_ret += float(reward)
        traj.append(np.array(env.agent_pos, dtype=np.float32).copy())

        if visualize and HAS_PYGAME and screen is not None:
            screen.fill((14, 16, 20))
            draw_navmesh_overlay(screen, env, world_to_screen)

            hint = recover_descent_path_world(env, max_len=512)
            if len(hint) >= 2:
                pygame.draw.lines(screen, (230, 180, 70), False, [world_to_screen(p) for p in hint], 2)

            if len(traj) >= 2:
                pygame.draw.lines(screen, (80, 220, 120), False, [world_to_screen(p) for p in traj], 3)

            tactical_target = final_info.get("tactical_target")
            if tactical_target is not None:
                pygame.draw.circle(screen, (120, 120, 240), world_to_screen(tactical_target), 4)

            pygame.draw.circle(screen, (230, 90, 90), world_to_screen(goal_pos), 6)
            pygame.draw.circle(screen, (80, 180, 250), world_to_screen(env.agent_pos), 5)

            d_geo = get_geodesic_distance(env)
            dist = float(d_geo) if d_geo is not None else float(np.linalg.norm(env.goal_pos - env.agent_pos))
            lines = [
                f"Step: {step + 1}/{max_steps}",
                f"Return: {ep_ret:.3f}",
                f"Dist: {dist:.2f}",
                f"Pos: ({env.agent_pos[0]:.1f}, {env.agent_height:.1f}, {env.agent_pos[1]:.1f})",
            ]
            y = 8
            for line in lines:
                surf = font.render(line, True, (220, 220, 220))
                screen.blit(surf, (8, y))
                y += 18

            pygame.display.flip()
            clock.tick(60)

        if terminated or truncated:
            break

    if save_csv_path is not None:
        try:
            np.savetxt(save_csv_path, np.stack(traj, axis=0), delimiter=",")
            print(f"[Saved] Trajectory -> {save_csv_path}")
        except Exception as exc:
            print(f"[Warn] Failed to save trajectory: {exc}")

    success = bool((final_info.get("reward_terms") or {}).get("success", 0))
    if success:
        outcome = "success"
    elif truncated:
        outcome = "timeout"
    elif bool(final_info.get("collided", False)):
        outcome = "blocked"
    else:
        outcome = "failed"

    print(f"[Eval] {outcome} | return={ep_ret:.3f}")
    return ep_ret, success, outcome, screen_bundle


def run_multiple_evaluations(env, actor, episodes=10, max_steps=None, scale=0.03, visualize=True, visualize_every=1, auto_quit=True, save_last_csv=None):
    returns = []
    successes = 0
    screen_bundle = None

    for ep in range(episodes):
        vis = visualize and ((ep % visualize_every) == 0)
        save_csv = save_last_csv if ep == episodes - 1 else None
        ret, succ, outcome, screen_bundle = evaluate_once(
            env,
            actor,
            max_steps=max_steps,
            scale=scale,
            screen_bundle=screen_bundle if vis else None,
            visualize=vis,
            save_csv_path=save_csv,
        )
        returns.append(ret)
        successes += int(succ)
        print(f"[Episode {ep + 1}/{episodes}] return={ret:.3f} outcome={outcome}")

    if visualize and HAS_PYGAME and screen_bundle and auto_quit:
        pygame.quit()

    avg_ret = float(np.mean(returns)) if returns else 0.0
    print(f"[Summary] episodes={episodes} success={successes} ({100.0 * successes / max(1, episodes):.1f}%) avg_return={avg_ret:.3f}")
    return returns


if __name__ == "__main__":
    import ENV
    from Model import GaussianPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path = "sac_actor_best.pth"
    env = ENV.make_env(seed=1)

    if not os.path.exists(actor_path):
        print(f"[WARN] {actor_path} not found. Train with Test.py first.")
        sys.exit(0)

    actor = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    run_multiple_evaluations(
        env,
        actor,
        episodes=50,
        scale=0.03,
        visualize=HAS_PYGAME,
        visualize_every=1,
        auto_quit=True,
        save_last_csv=str(Path("last_eval_traj.csv").resolve()),
    )
