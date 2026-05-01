import os
import sys
import argparse
from pathlib import Path

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

import torch


ROLE_NAMES = {-1: "none", 2: "base_move"}
HEURISTIC_SHORT = {"base_move_only": "bmove"}

ROLE_COLORS = {
    -1: (120, 120, 120),
    2: (160, 255, 160),
}

DETOUR_PATH_COLOR = (80, 255, 220)


def maybe_sample_agent_role_rules(env):
    min_agents = int(getattr(env, "random_agent_count_min", 0) or 0)
    max_agents = int(getattr(env, "random_agent_count_max", 0) or 0)
    if max_agents >= max(1, min_agents):
        count = int(np.random.randint(max(1, min_agents), max_agents + 1))
        chosen = ["base_move_only"] * count
        if hasattr(env, "configure_agent_group") and callable(env.configure_agent_group):
            env.configure_agent_group(chosen)
        else:
            env.agent_role_rules = list(chosen)
        setattr(env, "_current_agent_role_rule_sample", list(chosen))
        setattr(env, "_current_agent_role_rule_sample_index", None)
        return chosen
    pool = getattr(env, "agent_role_rule_pool", None)
    if not pool:
        return None
    idx = int(np.random.randint(0, len(pool)))
    chosen = [str(x).strip().lower() for x in pool[idx]]
    if hasattr(env, "configure_agent_group") and callable(env.configure_agent_group):
        env.configure_agent_group(chosen)
    else:
        env.agent_role_rules = list(chosen)
    setattr(env, "_current_agent_role_rule_sample", list(chosen))
    setattr(env, "_current_agent_role_rule_sample_index", idx)
    return chosen


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


from Model import GaussianPolicy, is_diverse_tactical_success

def build_env(**env_kwargs):
    try:
        from majestro_navmesh_env import MajestroNavMeshEnv
        return MajestroNavMeshEnv(**env_kwargs)
    except Exception as majestro_exc:
        try:
            import ENV
            if hasattr(ENV, "make_env") and callable(ENV.make_env):
                return ENV.make_env(**env_kwargs)
            raise RuntimeError("ENV module exists but make_env() is missing.")
        except Exception as env_exc:
            raise RuntimeError(
                "Failed to build evaluation env. Tried majestro_navmesh_env.MajestroNavMeshEnv and ENV.make_env().\n"
                f"majestro_navmesh_env error: {majestro_exc}\n"
                f"ENV error: {env_exc}"
            )


def reset_env_compat(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def recover_descent_path_world(env, start_pos, start_height=None, max_len=256):
    recover = getattr(env, "recover_fallback_path_world", None)
    if callable(recover):
        try:
            return recover(start_pos, start_height=start_height, max_len=max_len)
        except TypeError:
            try:
                return recover(start_pos)
            except Exception:
                pass

    geo = getattr(env, "_geo_map", None)
    pos_to_rc = getattr(env, "_pos_to_geo_rc", None)
    rc_to_world = getattr(env, "_grid_rc_to_world", None)
    goal_rc = getattr(env, "_geo_goal_rc", None)
    if geo is None or not callable(pos_to_rc) or not callable(rc_to_world):
        return []

    rows, cols = geo.shape
    start_r, start_c = pos_to_rc(start_pos)

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
    pts = [np.asarray(start_pos, dtype=np.float32).copy()]
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
    arr = np.asarray(obs_np, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    actions = np.zeros((arr.shape[0], 2), dtype=np.float32)
    sensor_ok = arr[:, -1] <= 0.5 if arr.shape[-1] > 0 else np.ones((arr.shape[0],), dtype=bool)
    idxs = np.where(sensor_ok)[0]
    if idxs.size > 0:
        s = torch.as_tensor(arr[idxs], dtype=torch.float32, device=next(actor.parameters()).device)
        actions[idxs] = actor.act_deterministic(s).cpu().numpy()
    return actions.reshape(-1) if squeeze else actions


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
    obs, info = reset_env_compat(env)
    actor.eval()

    start_pos = np.array(env.agent_pos, dtype=np.float32).copy()
    goal_pos = np.array(env.goal_pos, dtype=np.float32).copy()
    max_steps = int(max_steps or getattr(env, "max_steps", 300))

    agent_trajs = [[np.array(pos, dtype=np.float32).copy()] for pos in np.asarray(env.agent_positions, dtype=np.float32)]
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
    env_terminated = False
    env_truncated = False
    user_aborted = False

    for step in range(max_steps):
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    user_aborted = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    user_aborted = True

        if user_aborted:
            break

        action = policy_act(actor, obs)
        obs, reward, env_terminated, env_truncated, final_info = env.step(action)
        ep_ret += float(np.mean(np.asarray(reward, dtype=np.float32)))
        traj.append(np.array(env.agent_pos, dtype=np.float32).copy())
        for idx, pos in enumerate(np.asarray(env.agent_positions, dtype=np.float32)):
            if idx < len(agent_trajs):
                agent_trajs[idx].append(pos.copy())

        if visualize and HAS_PYGAME and screen is not None:
            screen.fill((14, 16, 20))
            draw_navmesh_overlay(screen, env, world_to_screen)

            role_ids = final_info.get("role_ids")
            if role_ids is None:
                role_ids = np.zeros((len(agent_trajs),), dtype=np.int32)
            else:
                role_ids = np.asarray(role_ids).reshape(-1)

            sense_radius = float(getattr(env, "sense_radius", 0.0))
            sense_px = max(1, int(round(sense_radius * scale)))
            if sense_px > 0:
                for idx, pos in enumerate(np.asarray(env.agent_positions, dtype=np.float32)):
                    role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                    color = ROLE_COLORS.get(role_id, (160, 160, 160))
                    pygame.draw.circle(screen, color, world_to_screen(pos), sense_px, 1)

            sensor_fail_codes = final_info.get("sensor_fail_code")
            fail_arr = None if sensor_fail_codes is None else np.asarray(sensor_fail_codes, dtype=np.float32).reshape(-1)
            if fail_arr is not None:
                for idx, pos in enumerate(np.asarray(env.agent_positions, dtype=np.float32)):
                    if idx >= len(fail_arr) or fail_arr[idx] <= 0.5:
                        continue
                    start_height = None
                    if idx < len(env.agent_heights):
                        start_height = float(env.agent_heights[idx])
                    path = recover_descent_path_world(env, pos, start_height=start_height, max_len=256)
                    if len(path) >= 2:
                        pygame.draw.lines(screen, DETOUR_PATH_COLOR, False, [world_to_screen(p) for p in path], 2)

            for idx, points in enumerate(agent_trajs):
                if len(points) < 2:
                    continue
                role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                color = ROLE_COLORS.get(role_id, (160, 160, 160))
                pygame.draw.lines(screen, color, False, [world_to_screen(p) for p in points], 2)

            tactical_target = final_info.get("tactical_target")
            if tactical_target is not None:
                tactical_target = np.asarray(tactical_target, dtype=np.float32)
                if tactical_target.ndim == 1:
                    pygame.draw.circle(screen, (120, 120, 240), world_to_screen(tactical_target), 4)
                else:
                    pygame.draw.circle(screen, (120, 120, 240), world_to_screen(tactical_target[0]), 4)
                    for other_target in tactical_target[1:]:
                        pygame.draw.circle(screen, (110, 110, 170), world_to_screen(other_target), 3)

            role_targets = final_info.get("role_targets")
            if role_targets is not None:
                for idx, role_target in enumerate(np.asarray(role_targets, dtype=np.float32)):
                    role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                    color = ROLE_COLORS.get(role_id, (170, 110, 110))
                    pygame.draw.circle(screen, color, world_to_screen(role_target), 3, 1)

            agent_positions = final_info.get("agent_positions")
            if agent_positions is not None:
                for idx, other in enumerate(np.asarray(agent_positions, dtype=np.float32)):
                    role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                    color = ROLE_COLORS.get(role_id, (200, 140, 70))
                    sx, sy = world_to_screen(other)
                    pygame.draw.circle(screen, color, (sx, sy), 4)
                    label = f"A{idx}"
                    surf = font.render(label, True, color)
                    screen.blit(surf, (sx + 6, sy - 10))

            pygame.draw.circle(screen, (230, 90, 90), world_to_screen(goal_pos), 6)
            pygame.draw.circle(screen, (255, 255, 255), world_to_screen(env.agent_pos), 5, 1)

            dist = float(np.linalg.norm(env.goal_pos - env.agent_pos))
            lines = [
                f"Step: {step + 1}/{max_steps}",
                f"Return: {ep_ret:.3f}",
                f"Dist: {dist:.2f}",
                f"Pos: ({env.agent_pos[0]:.1f}, {env.agent_height:.1f}, {env.agent_pos[1]:.1f})",
            ]
            if role_ids is not None:
                role_ids_arr = np.asarray(role_ids, dtype=np.int32).reshape(-1)
                none_count = int(np.sum(role_ids_arr == -1))
                base_move_count = int(np.sum(role_ids_arr == 2))
                lines.append(f"Roles: none={none_count} base_move={base_move_count}")
            y = 8
            for line in lines:
                surf = font.render(line, True, (220, 220, 220))
                screen.blit(surf, (8, y))
                y += 18

            pygame.display.flip()
            clock.tick(60)

        if env_terminated or env_truncated:
            break

    if save_csv_path is not None:
        try:
            np.savetxt(save_csv_path, np.stack(traj, axis=0), delimiter=",")
            print(f"[Saved] Trajectory -> {save_csv_path}")
        except Exception as exc:
            print(f"[Warn] Failed to save trajectory: {exc}")

    success = bool(is_diverse_tactical_success(final_info))
    if user_aborted:
        outcome = "aborted"
    elif success:
        outcome = "success"
    elif env_truncated:
        outcome = "timeout"
    elif np.any(np.asarray(final_info.get("collided", False))):
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
        sampled_rules = maybe_sample_agent_role_rules(env)
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
        rule_summary = "" if sampled_rules is None else f" agents={len(sampled_rules)}"
        print(f"[Episode {ep + 1}/{episodes}] return={ret:.3f} outcome={outcome}{rule_summary}")

        if outcome == "aborted":
            print("[Info] Evaluation stopped by user.")
            break

    if visualize and HAS_PYGAME and screen_bundle and auto_quit:
        pygame.quit()

    avg_ret = float(np.mean(returns)) if returns else 0.0
    print(f"[Summary] episodes={len(returns)} success={successes} ({100.0 * successes / max(1, len(returns)):.1f}%) avg_return={avg_ret:.3f}")
    return returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate shared-policy SAC on Majestro NavMesh.")
    ap.add_argument("--actor-path", type=str, default="sac_actor_best.pth")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--no-visualize", action="store_true", default=False)
    ap.add_argument("--move-step-size", type=float, default=120.0)
    ap.add_argument("--tactical-target-radius", type=float, default=600.0)
    ap.add_argument("--num-other-agents", type=int, default=4)
    ap.add_argument("--random-agent-count-min", type=int, default=3)
    ap.add_argument("--random-agent-count-max", type=int, default=20)
    ap.add_argument("--observed-other-agents", type=int, default=3)
    ap.add_argument("--agent-radius", type=float, default=90.0)
    ap.add_argument("--sense-radius", type=float, default=1000.0)
    ap.add_argument("--goal-spawn-min-scale", type=float, default=4.0)
    ap.add_argument("--agent-spawn-min-scale", type=float, default=2.0)
    ap.add_argument("--agent-spawn-max-scale", type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path = args.actor_path
    num_agents = 1 + int(args.num_other_agents)
    agent_role_rules = ["base_move_only"] * num_agents

    env = build_env(
        seed=args.seed,
        move_step_size=args.move_step_size,
        tactical_target_radius=args.tactical_target_radius,
        num_other_agents=args.num_other_agents,
        observed_other_agents=args.observed_other_agents,
        agent_radius=args.agent_radius,
        sense_radius=args.sense_radius,
        goal_spawn_min_scale=args.goal_spawn_min_scale,
        agent_spawn_min_scale=args.agent_spawn_min_scale,
        agent_spawn_max_scale=args.agent_spawn_max_scale,
        role_rule="base_move_only",
        agent_role_rules=agent_role_rules,
    )
    env.random_agent_count_min = int(args.random_agent_count_min)
    env.random_agent_count_max = int(max(args.random_agent_count_min, args.random_agent_count_max))
    print(f"[AGENT-COUNT] random total agents per episode: {env.random_agent_count_min}..{env.random_agent_count_max}")

    if not os.path.exists(actor_path):
        print(f"[WARN] {actor_path} not found. Train with Test.py first.")
        sys.exit(0)

    obs_dim = int(getattr(env, "single_agent_obs_dim", env.observation_space.shape[-1]))
    act_dim = int(getattr(env, "single_agent_act_dim", env.action_space.shape[-1]))
    state_obj = torch.load(actor_path, map_location=device, weights_only=False)
    if state_obj.get("format") != "base_move_actor":
        raise RuntimeError("Expected base_move_actor checkpoint.")
    actor = GaussianPolicy(obs_dim, act_dim).to(device)
    actor.load_state_dict(state_obj["actor"])
    actor.eval()

    run_multiple_evaluations(
        env,
        actor,
        episodes=args.episodes,
        scale=args.scale,
        visualize=(HAS_PYGAME and (not args.no_visualize)),
        visualize_every=1,
        auto_quit=True,
        save_last_csv=str(Path("last_eval_traj.csv").resolve()),
    )
