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


ROLE_RULE_ALIASES = {
    "mdps": "melee_dps",
    "melee_dps": "melee_dps",
    "meleedps": "melee_dps",
    "rdps": "ranged_dps",
    "ranged_dps": "ranged_dps",
    "rangeddps": "ranged_dps",
    "fix": "fixed",
    "fixed": "fixed",
}


ROLE_NAMES = {
    -1: "none",
    0: "front",
    1: "cover",
    2: "base",
    3: "surround",
    4: "kiting",
}

HEURISTIC_SHORT = {
    "fixed": "fix",
    "melee_dps": "mdps",
    "meleedps": "mdps",
    "ranged_dps": "rdps",
    "rangeddps": "rdps",
}

ROLE_COLORS = {
    -1: (120, 120, 120),
    0: (255, 110, 110),
    1: (180, 140, 255),
    2: (160, 255, 160),
    3: (255, 140, 200),
    4: (255, 170, 110),
}

DETOUR_PATH_COLOR = (80, 255, 220)
SURROUND_RING_COLOR = (64, 128, 255)
RANDOM_TARGET_COLOR = (255, 230, 90)
FRONT_RING_COLOR = (255, 96, 96)
KITING_RING_COLOR = (255, 170, 64)
SIDEBAR_WIDTH = 540

# Code-level defaults. Change these if you want to switch sources without CLI args.
DEFAULT_ACTOR_PATH = "sac_actor_last.pth"
DEFAULT_ACTOR_SOURCE = "latest"  # "latest" | "single-best"


def _normalize_rule_name(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Empty role rule is not allowed.")
    return ROLE_RULE_ALIASES.get(key, key)


def parse_agent_role_rules(text: str | None, num_agents: int):
    if not text:
        return None
    parts = [_normalize_rule_name(part) for part in str(text).split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) > num_agents:
        raise ValueError(f"agent_role_rules length must be <= num_agents ({num_agents}), got {len(parts)}")
    if len(parts) < num_agents:
        parts.extend([parts[-1]] * (num_agents - len(parts)))
    return parts


def parse_agent_role_rule_pool(text: str | None, num_agents: int):
    if not text:
        return None
    pool = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        rules = [_normalize_rule_name(part) for part in chunk.split(",") if part.strip()]
        if rules:
            pool.append(rules)
    return pool or None


def maybe_sample_agent_role_rules(env):
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


def apply_agent_role_rules(env, chosen_rules, chosen_index=None):
    chosen = [str(x).strip().lower() for x in chosen_rules]
    if hasattr(env, "configure_agent_group") and callable(env.configure_agent_group):
        env.configure_agent_group(chosen)
    else:
        env.agent_role_rules = list(chosen)
    setattr(env, "_current_agent_role_rule_sample", list(chosen))
    setattr(env, "_current_agent_role_rule_sample_index", chosen_index)
    return chosen


def choose_agent_role_rules(env, selection_state=None):
    pool = getattr(env, "agent_role_rule_pool", None)
    if not pool:
        return None
    if selection_state is None:
        return maybe_sample_agent_role_rules(env)
    selected_index = selection_state.get("selected_index")
    if selected_index is None:
        chosen = maybe_sample_agent_role_rules(env)
        selection_state["active_index"] = getattr(env, "_current_agent_role_rule_sample_index", None)
        return chosen
    idx = int(max(0, min(len(pool) - 1, int(selected_index))))
    selection_state["active_index"] = idx
    return apply_agent_role_rules(env, pool[idx], chosen_index=idx)


def _format_rule_set_label(rules):
    return ",".join(HEURISTIC_SHORT.get(str(rule), str(rule)[:4]) for rule in rules)


def make_world_to_screen(bounds_min, bounds_max, scale, sidebar_width=0):
    min_x, min_z = float(bounds_min[0]), float(bounds_min[1])
    max_x, max_z = float(bounds_max[0]), float(bounds_max[1])
    map_width = max(1, int((max_x - min_x) * scale))
    width = map_width + max(0, int(sidebar_width))
    height = max(1, int((max_z - min_z) * scale))

    def world_to_screen(p):
        x, z = float(p[0]), float(p[1])
        sx = int((x - min_x) * scale)
        sy = int((max_z - z) * scale)
        return sx, sy

    return width, height, world_to_screen, map_width


try:
    from Model import (
        GaussianPolicy,
        is_diverse_tactical_success,
        ROLE_IDS,
        POLICY_ROLE_IDS,
        role_name,
        get_env_role_ids,
        role_policy_actions,
    )
except Exception:
    from Model import GaussianPolicy

    def is_diverse_tactical_success(info):
        role_ids = np.asarray(info.get("role_ids", []), dtype=np.int32).reshape(-1)
        success_mask = np.asarray(info.get("success_mask", []), dtype=bool).reshape(-1)
        front = False
        cover = False
        surround = False
        for role_id, success in zip(role_ids, success_mask):
            if not bool(success):
                continue
            if int(role_id) == 0:
                front = True
            elif int(role_id) == 1:
                cover = True
            elif int(role_id) == 3:
                surround = True
        return bool(front and cover and surround)

    ROLE_IDS = (0, 1, 2, 3, 4)
    POLICY_ROLE_IDS = (0, 1, 3, 4)

    def role_name(role_id):
        return {-1: "none", 0: "front", 1: "cover", 2: "base_move", 3: "surround", 4: "kiting"}.get(int(role_id), f"role_{int(role_id)}")

    def get_env_role_ids(env, count):
        role_ids = getattr(env, "agent_role_ids", None)
        if role_ids is None:
            return np.zeros((count,), dtype=np.int32)
        return np.asarray(role_ids, dtype=np.int32).reshape(-1)[:count]

    @torch.no_grad()
    def role_policy_actions(role_bundles, obs_arr, role_ids_arr, deterministic=True):
        actions = np.zeros((obs_arr.shape[0], 2), dtype=np.float32)
        obs_arr = np.asarray(obs_arr, dtype=np.float32)
        sensor_ok = obs_arr[:, -1] <= 0.5 if obs_arr.ndim >= 2 and obs_arr.shape[-1] > 0 else np.ones((obs_arr.shape[0],), dtype=bool)
        for role_id in ROLE_IDS:
            idxs = np.where((role_ids_arr == role_id) & sensor_ok)[0]
            if idxs.size == 0:
                continue
            if int(role_id) not in role_bundles:
                continue
            actor = role_bundles[int(role_id)]["actor"]
            device = next(actor.parameters()).device
            s = torch.as_tensor(obs_arr[idxs], dtype=torch.float32, device=device)
            if deterministic:
                a = actor.act_deterministic(s).cpu().numpy()
            else:
                a, _ = actor.sample(s)
                a = a.detach().cpu().numpy()
            actions[idxs] = a
        return actions

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


def load_actor_state_map(actor_source: str, actor_path: str, role_ids) -> dict:
    actor_source = str(actor_source).strip().lower()
    if actor_source == "latest":
        state_obj = torch.load(actor_path, map_location="cpu", weights_only=False)
        if state_obj.get("format") != "multi_role_actor":
            raise RuntimeError("Expected multi_role_actor checkpoint for actor_source='latest'.")
        actors = state_obj.get("actors")
        if not isinstance(actors, dict):
            raise RuntimeError("Latest checkpoint is missing an 'actors' dict.")
        return actors
    if actor_source == "single-best":
        actor_states = {}
        actor_root = Path(actor_path)
        for role_id in role_ids:
            role_key = role_name(role_id)
            role_actor_path = actor_root.with_name(f"{actor_root.stem}_{role_key}{actor_root.suffix or '.pth'}")
            if not role_actor_path.exists():
                fallback_root = actor_root.with_name(actor_root.name.replace("_last", "_best"))
                fallback_role_actor_path = fallback_root.with_name(f"{fallback_root.stem}_{role_key}{fallback_root.suffix or '.pth'}")
                if fallback_role_actor_path.exists():
                    role_actor_path = fallback_role_actor_path
                else:
                    raise RuntimeError(
                        f"Single-best actor checkpoint not found for role '{role_key}': "
                        f"{role_actor_path} (fallback tried: {fallback_role_actor_path})"
                    )
            state_obj = torch.load(role_actor_path, map_location="cpu", weights_only=False)
            if state_obj.get("format") != "single_role_actor":
                raise RuntimeError(f"Expected single_role_actor checkpoint for role '{role_key}', got {state_obj.get('format')!r}")
            actor_state = state_obj.get("actor")
            if actor_state is None:
                raise RuntimeError(f"Single-best checkpoint for role '{role_key}' is missing actor weights.")
            actor_states[role_key] = actor_state
        return actor_states
    raise RuntimeError(f"Unsupported actor_source: {actor_source!r}")


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
def policy_act(role_bundles, env, obs_np):
    arr = np.asarray(obs_np, dtype=np.float32)
    if arr.ndim == 1:
        role_ids_arr = get_env_role_ids(env, 1)
        return role_policy_actions(role_bundles, arr.reshape(1, -1), role_ids_arr, deterministic=True).reshape(-1)
    role_ids_arr = get_env_role_ids(env, arr.shape[0])
    return role_policy_actions(role_bundles, arr, role_ids_arr, deterministic=True)


def _sample_random_detour_target(env, pos, rng, max_tries=64):
    sense_radius = max(1.0, float(getattr(env, "sense_radius", 0.0)))
    nearest_valid = getattr(env, "_nearest_valid_point", None)
    if not callable(nearest_valid):
        return np.asarray(pos, dtype=np.float32).copy()
    for _ in range(max(1, int(max_tries))):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = float(sense_radius * np.sqrt(rng.uniform(0.0, 1.0)))
        cand = np.asarray(pos, dtype=np.float32) + np.array(
            [np.cos(theta) * radius, np.sin(theta) * radius],
            dtype=np.float32,
        )
        snapped = nearest_valid(cand, max_radius=8)
        if snapped is None:
            continue
        snapped_pos = np.asarray(snapped[0], dtype=np.float32).copy()
        if float(np.linalg.norm(snapped_pos - pos)) >= max(8.0, float(getattr(env, "agent_radius", 0.0)) * 0.5):
            return snapped_pos
    return np.asarray(pos, dtype=np.float32).copy()


def _build_detour_path_to_target(env, start_pos, target_pos, start_height=None, target_height=None, max_hops=32):
    path = [np.asarray(start_pos, dtype=np.float32).reshape(-1)[:2].copy()]
    detour_to_target = getattr(env, "_detour_next_waypoint_to_target", None)
    sample_height = getattr(env, "_sample_height", None)
    if not callable(detour_to_target):
        path.append(np.asarray(target_pos, dtype=np.float32).reshape(-1)[:2].copy())
        return path

    cur = path[0].copy()
    cur_height = float(getattr(env, "goal_height", 0.0) if start_height is None else start_height)
    tgt = np.asarray(target_pos, dtype=np.float32).reshape(-1)[:2].copy()
    tgt_height = float(cur_height if target_height is None else target_height)
    seen = set()
    reach_tol = max(float(getattr(env, "_grid_cell_size", 1.0)) * 0.75, 8.0)

    for _ in range(max(1, int(max_hops))):
        key = (round(float(cur[0]), 2), round(float(cur[1]), 2))
        if key in seen:
            break
        seen.add(key)
        waypoint = detour_to_target(cur, tgt, height=cur_height, target_height=tgt_height)
        if waypoint is None:
            break
        waypoint = np.asarray(waypoint, dtype=np.float32).reshape(-1)[:2]
        if float(np.linalg.norm(waypoint - cur)) <= 1e-3:
            break
        path.append(waypoint.copy())
        cur = waypoint
        if callable(sample_height):
            h = sample_height(cur)
            if h is not None:
                cur_height = float(h)
        if float(np.linalg.norm(tgt - cur)) <= reach_tol:
            break

    if float(np.linalg.norm(path[-1] - tgt)) > 1e-3:
        path.append(tgt.copy())
    return path


def update_random_moving_goal(env, rng, target_state):
    current_goal = np.asarray(getattr(env, "goal_pos", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)[:2]

    sample_height = getattr(env, "_sample_height", None)
    world_to_grid = getattr(env, "_world_to_grid_rc", None)
    compute_geo = getattr(env, "_compute_geodesic_map", None)
    detour_with_progress = getattr(env, "_detour_next_waypoint_with_min_progress", None)

    goal_height = float(getattr(env, "goal_height", 0.0))
    move_step = max(1.0, float(getattr(env, "step_size", 1.0)))
    min_progress = max(float(getattr(env, "_grid_cell_size", 1.0)) * 0.75, move_step * 0.35)
    reach_tol = max(float(getattr(env, "_grid_cell_size", 1.0)) * 0.75, move_step * 0.5, 8.0)
    goal_destination = target_state.get("goal_destination")
    if goal_destination is not None:
        goal_destination = np.asarray(goal_destination, dtype=np.float32).reshape(-1)[:2]
        if float(np.linalg.norm(goal_destination - current_goal)) <= reach_tol:
            goal_destination = None
    if goal_destination is None:
        goal_destination = _sample_random_detour_target(env, current_goal, rng)
        target_state["goal_destination"] = goal_destination.copy()

    dest_height = goal_height
    if callable(sample_height):
        dest_sample_h = sample_height(goal_destination)
        if dest_sample_h is not None:
            dest_height = float(dest_sample_h)
    target_state["goal_destination_height"] = float(dest_height)

    next_goal = current_goal.copy()
    next_goal_height = goal_height
    fallback_direct = False
    if callable(detour_with_progress):
        detour_result = detour_with_progress(
            current_goal,
            goal_destination,
            min_progress=min_progress,
            height=float(goal_height),
            target_height=dest_height,
        )
        if detour_result is not None:
            waypoint, waypoint_height = detour_result
            waypoint = np.asarray(waypoint, dtype=np.float32).reshape(-1)[:2]
            delta = waypoint - current_goal
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                if dist > move_step:
                    next_goal = current_goal + (delta / dist) * move_step
                else:
                    next_goal = waypoint
                next_goal_height = float(waypoint_height)
        else:
            delta = goal_destination - current_goal
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                if dist > move_step:
                    next_goal = current_goal + (delta / dist) * move_step
                else:
                    next_goal = goal_destination.copy()
                if callable(sample_height):
                    step_h = sample_height(next_goal)
                    if step_h is not None:
                        next_goal_height = float(step_h)
    else:
        fallback_direct = True

    if fallback_direct:
        delta = goal_destination - current_goal
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6:
            if dist > move_step:
                next_goal = current_goal + (delta / dist) * move_step
            else:
                next_goal = goal_destination.copy()
            if callable(sample_height):
                step_h = sample_height(next_goal)
                if step_h is not None:
                    next_goal_height = float(step_h)

    env.goal_pos = np.asarray(next_goal, dtype=np.float32)
    env.goal_height = float(next_goal_height)
    target_state["goal"] = env.goal_pos.copy()
    target_state["goal_path"] = _build_detour_path_to_target(
        env,
        env.goal_pos,
        goal_destination,
        start_height=env.goal_height,
        target_height=dest_height,
    )
    if callable(sample_height):
        h = sample_height(env.goal_pos)
        if h is not None:
            env.goal_height = float(h)
    if callable(world_to_grid) and callable(compute_geo):
        goal_rc = world_to_grid(env.goal_pos)
        env._geo_goal_rc = goal_rc
        env._geo_map = compute_geo(goal_rc)
    if hasattr(env, "_update_role_targets") and callable(env._update_role_targets):
        env._update_role_targets()
    if hasattr(env, "current_detour_waypoints"):
        env.current_detour_waypoints[:] = np.nan
    if hasattr(env, "current_detour_waypoint_heights"):
        env.current_detour_waypoint_heights[:] = np.nan


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


def evaluate_once(env, role_bundles, max_steps=None, scale=0.03, screen_bundle=None, visualize=True, save_csv_path=None,
                  random_detour_mode=False, random_target_state=None, selection_state=None):
    obs, info = reset_env_compat(env)
    for bundle in role_bundles.values():
        bundle["actor"].eval()

    start_pos = np.array(env.agent_pos, dtype=np.float32).copy()
    max_steps = int(max_steps or getattr(env, "max_steps", 300))

    agent_trajs = [[np.array(pos, dtype=np.float32).copy()] for pos in np.asarray(env.agent_positions, dtype=np.float32)]
    traj = [start_pos.copy()]
    screen = clock = font = None
    world_to_screen = None

    if visualize and HAS_PYGAME:
        if screen_bundle is None:
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "120,160")
            pygame.init()
            sidebar_width = SIDEBAR_WIDTH if selection_state is not None else 0
            width, height, world_to_screen, map_width = make_world_to_screen(env.bounds_min, env.bounds_max, scale, sidebar_width=sidebar_width)
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("ModelTest - Majestro NavMesh")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("consolas", 16)
            screen_bundle = (screen, clock, font, world_to_screen, map_width)
        else:
            screen, clock, font, world_to_screen, map_width = screen_bundle
    else:
        map_width = None

    ep_ret = 0.0
    final_info = {}
    env_terminated = False
    env_truncated = False
    user_aborted = False
    restart_requested = False

    for step in range(max_steps):
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    user_aborted = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    user_aborted = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and selection_state is not None:
                    mouse_pos = getattr(event, "pos", None)
                    if mouse_pos is not None:
                        for button in selection_state.get("buttons", []):
                            rect = button.get("rect")
                            if rect is not None and rect.collidepoint(mouse_pos):
                                kind = button.get("kind", "preset")
                                if kind == "toggle_moving_goal":
                                    new_state = not bool(getattr(env, "moving_goal_enabled", False))
                                    env.moving_goal_enabled = new_state
                                    selection_state["moving_goal_enabled"] = new_state
                                else:
                                    selected_index = button.get("index")
                                    selection_state["selected_index"] = selected_index
                                    selection_state["restart_requested"] = True
                                    restart_requested = True
                                break

        if user_aborted or restart_requested:
            break

        if random_detour_mode:
            action = np.zeros((len(np.asarray(getattr(env, "agent_positions", [env.agent_pos]))), 2), dtype=np.float32)
            if action.shape[0] == 1:
                action = action.reshape(-1)
        else:
            action = policy_act(role_bundles, env, obs)
        obs, reward, env_terminated, env_truncated, final_info = env.step(action)
        ep_ret += float(np.mean(np.asarray(reward, dtype=np.float32)))
        traj.append(np.array(env.agent_pos, dtype=np.float32).copy())
        for idx, pos in enumerate(np.asarray(env.agent_positions, dtype=np.float32)):
            if idx < len(agent_trajs):
                agent_trajs[idx].append(pos.copy())

        if visualize and HAS_PYGAME and screen is not None:
            screen.fill((14, 16, 20))
            draw_navmesh_overlay(screen, env, world_to_screen)
            current_goal_pos = np.asarray(final_info.get("goal_pos", getattr(env, "goal_pos", np.zeros((2,), dtype=np.float32))), dtype=np.float32).reshape(-1)[:2]
            current_goal_destination = np.asarray(final_info.get("goal_destination", getattr(env, "goal_destination", current_goal_pos)), dtype=np.float32).reshape(-1)[:2]
            current_goal_path = final_info.get("goal_path", getattr(env, "goal_path", []))

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

            surround_ring_px = None
            front_ring_px = None
            kiting_ring_px = None
            front_mask = np.asarray(role_ids, dtype=np.int32).reshape(-1) == 0
            if np.any(front_mask):
                front_radius = float(getattr(env, "front_success_radius", getattr(env, "success_radius", 0.0) * 1.15))
                front_ring_px = max(10, int(round(front_radius * scale)))
            surround_mask = np.asarray(role_ids, dtype=np.int32).reshape(-1) == 3
            if np.any(surround_mask):
                surround_radius = max(
                    float(getattr(env, "success_radius", 0.0)) * 1.6,
                    float(getattr(env, "agent_radius", 0.0)) * 2.2,
                )
                surround_ring_px = max(12, int(round(surround_radius * scale)))
            kiting_mask = np.asarray(role_ids, dtype=np.int32).reshape(-1) == 4
            if np.any(kiting_mask):
                kiting_ring_px = max(12, int(round(700.0 * scale)))

            sensor_fail_codes = final_info.get("sensor_fail_code")
            fail_arr = None if sensor_fail_codes is None else np.asarray(sensor_fail_codes, dtype=np.float32).reshape(-1)
            agent_role_rules = final_info.get("agent_role_rules")
            if agent_role_rules is None:
                default_rule = getattr(env, "role_rule", "fixed")
                agent_role_rules = [default_rule] * len(agent_trajs)
            else:
                agent_role_rules = list(agent_role_rules)
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
                    tx, ty = world_to_screen(tactical_target)
                    pygame.draw.circle(screen, (120, 120, 240), (tx, ty), 4)
                    surf = font.render("T", True, (120, 120, 240))
                    screen.blit(surf, (tx + 5, ty - 10))
                else:
                    for idx, target in enumerate(tactical_target):
                        tx, ty = world_to_screen(target)
                        role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                        if role_id == 4:
                            pygame.draw.circle(screen, (120, 120, 240), (tx, ty), 4)
                            surf = font.render("T", True, (120, 120, 240))
                            screen.blit(surf, (tx + 5, ty - 10))
                        else:
                            color = (120, 120, 240) if idx == 0 else (110, 110, 170)
                            radius = 4 if idx == 0 else 3
                            pygame.draw.circle(screen, color, (tx, ty), radius)

            if np.all(np.isfinite(current_goal_pos)):
                pygame.draw.circle(screen, RANDOM_TARGET_COLOR, world_to_screen(current_goal_pos), 7, 2)
            if np.all(np.isfinite(current_goal_destination)):
                pygame.draw.circle(screen, (120, 255, 120), world_to_screen(current_goal_destination), 6, 2)
            if current_goal_path:
                goal_path = [world_to_screen(p) for p in current_goal_path if np.all(np.isfinite(np.asarray(p, dtype=np.float32)))]
                if len(goal_path) >= 2:
                    pygame.draw.lines(screen, (120, 255, 120), False, goal_path, 2)

            role_targets = final_info.get("role_targets")
            if role_targets is not None:
                for idx, role_target in enumerate(np.asarray(role_targets, dtype=np.float32)):
                    role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                    color = ROLE_COLORS.get(role_id, (170, 110, 110))
                    rx, ry = world_to_screen(role_target)
                    pygame.draw.circle(screen, color, (rx, ry), 3, 1)
                    if role_id == 4:
                        surf = font.render("K", True, color)
                        screen.blit(surf, (rx + 5, ry + 2))

            if front_ring_px is not None:
                pygame.draw.circle(screen, FRONT_RING_COLOR, world_to_screen(current_goal_pos), front_ring_px, 2)
            if surround_ring_px is not None:
                pygame.draw.circle(screen, SURROUND_RING_COLOR, world_to_screen(current_goal_pos), surround_ring_px, 3)
            if kiting_ring_px is not None:
                pygame.draw.circle(screen, KITING_RING_COLOR, world_to_screen(current_goal_pos), kiting_ring_px, 2)
            pygame.draw.circle(screen, (255, 255, 255), world_to_screen(env.agent_pos), 5, 1)
            pygame.draw.circle(screen, (230, 90, 90), world_to_screen(current_goal_pos), 6)

            agent_positions = final_info.get("agent_positions")
            if agent_positions is not None:
                for idx, other in enumerate(np.asarray(agent_positions, dtype=np.float32)):
                    role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
                    color = ROLE_COLORS.get(role_id, (200, 140, 70))
                    sx, sy = world_to_screen(other)
                    pygame.draw.circle(screen, color, (sx, sy), 4)
                    rule_name = str(agent_role_rules[idx]) if idx < len(agent_role_rules) else str(getattr(env, "role_rule", "fixed"))
                    rule_short = HEURISTIC_SHORT.get(rule_name, rule_name[:3])
                    role_label = ROLE_NAMES.get(role_id, str(role_id))
                    label = f"{rule_short}->{role_label}"
                    surf = font.render(label, True, color)
                    screen.blit(surf, (sx + 6, sy - 10))

            dist = float(np.linalg.norm(env.goal_pos - env.agent_pos))
            lines = [
                f"Step: {step + 1}/{max_steps}",
                f"Return: {ep_ret:.3f}",
                f"Dist: {dist:.2f}",
                f"Pos: ({env.agent_pos[0]:.1f}, {env.agent_height:.1f}, {env.agent_pos[1]:.1f})",
            ]
            lines.append(f"Goal: ({current_goal_pos[0]:.1f}, {current_goal_pos[1]:.1f})")
            lines.append(f"GoalDest: ({current_goal_destination[0]:.1f}, {current_goal_destination[1]:.1f})")
            if role_ids is not None:
                role_labels = [ROLE_NAMES.get(int(r), str(int(r))) for r in role_ids]
                lines.append(f"Roles: {', '.join(role_labels)}")
            y = 8
            for line in lines:
                surf = font.render(line, True, (220, 220, 220))
                screen.blit(surf, (8, y))
                y += 18

            if selection_state is not None:
                buttons = []
                panel_w = max(280, SIDEBAR_WIDTH - 20)
                panel_x = (map_width if map_width is not None else screen.get_width() - SIDEBAR_WIDTH) + 10
                panel_y = 12
                preset_count = 1 + len(selection_state.get("pool", []))
                panel_h = 56 + 24 * preset_count
                pygame.draw.rect(screen, (24, 28, 36), pygame.Rect(panel_x, panel_y, panel_w, panel_h))
                pygame.draw.rect(screen, (70, 78, 92), pygame.Rect(panel_x, panel_y, panel_w, panel_h), 1)
                title = font.render("Viewer controls", True, (220, 220, 220))
                screen.blit(title, (panel_x + 8, panel_y + 6))
                toggle_rect = pygame.Rect(panel_x + 8, panel_y + 28, panel_w - 16, 20)
                moving_goal_enabled = bool(getattr(env, "moving_goal_enabled", False))
                selection_state["moving_goal_enabled"] = moving_goal_enabled
                toggle_fill = (52, 88, 62) if moving_goal_enabled else (72, 44, 44)
                toggle_text = "Moving Goal: ON" if moving_goal_enabled else "Moving Goal: OFF"
                pygame.draw.rect(screen, toggle_fill, toggle_rect)
                pygame.draw.rect(screen, (92, 96, 108), toggle_rect, 1)
                toggle_surf = font.render(toggle_text, True, (235, 235, 235))
                screen.blit(toggle_surf, (toggle_rect.x + 6, toggle_rect.y + 2))
                buttons.append({"rect": toggle_rect, "kind": "toggle_moving_goal"})
                subtitle = font.render("Rule presets: click to restart with selected preset", True, (220, 220, 220))
                screen.blit(subtitle, (panel_x + 8, panel_y + 54))
                active_index = selection_state.get("active_index")
                selected_index = selection_state.get("selected_index")
                button_y = panel_y + 76
                button_specs = [(None, "RND random")] + [
                    (idx, f"{idx + 1}. {_format_rule_set_label(rules)}")
                    for idx, rules in enumerate(selection_state.get("pool", []))
                ]
                for button_index, (preset_index, label) in enumerate(button_specs):
                    rect = pygame.Rect(panel_x + 8, button_y + 24 * button_index, panel_w - 16, 20)
                    is_selected = selected_index == preset_index
                    is_active = active_index == preset_index
                    fill = (48, 60, 82) if is_selected else (34, 38, 46)
                    border = (110, 180, 255) if is_active else (82, 88, 98)
                    text_color = (230, 240, 255) if is_selected or is_active else (205, 205, 205)
                    pygame.draw.rect(screen, fill, rect)
                    pygame.draw.rect(screen, border, rect, 1)
                    label_prefix = "* " if is_active else "  "
                    surf = font.render(f"{label_prefix}{label}", True, text_color)
                    screen.blit(surf, (rect.x + 6, rect.y + 2))
                    buttons.append({"rect": rect, "index": preset_index, "kind": "preset"})
                selection_state["buttons"] = buttons

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
    elif restart_requested or bool(selection_state is not None and selection_state.get("restart_requested")):
        if selection_state is not None:
            selection_state["restart_requested"] = False
        outcome = "rerun"
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


def run_multiple_evaluations(env, role_bundles, episodes=10, max_steps=None, scale=0.03, visualize=True, visualize_every=1,
                             auto_quit=True, save_last_csv=None, random_detour_mode=False):
    returns = []
    successes = 0
    screen_bundle = None
    pool = getattr(env, "agent_role_rule_pool", None)
    selection_state = {
        "pool": [list(rules) for rules in pool] if pool else [],
        "selected_index": 0 if pool else None,
        "active_index": None,
        "buttons": [],
        "restart_requested": False,
        "moving_goal_enabled": bool(getattr(env, "moving_goal_enabled", False)),
    }

    ep = 0
    while ep < episodes:
        sampled_rules = choose_agent_role_rules(env, selection_state=selection_state)
        vis = visualize and ((ep % visualize_every) == 0)
        save_csv = save_last_csv if ep == episodes - 1 else None
        ret, succ, outcome, screen_bundle = evaluate_once(
            env,
            role_bundles,
            max_steps=max_steps,
            scale=scale,
            screen_bundle=screen_bundle if vis else None,
            visualize=vis,
            save_csv_path=save_csv,
            random_detour_mode=random_detour_mode,
            selection_state=selection_state,
        )
        if outcome == "rerun":
            print("[Info] Restarting episode with selected rule preset.")
            continue
        returns.append(ret)
        successes += int(succ)
        rule_summary = "" if sampled_rules is None else f" agents={len(sampled_rules)} rules={','.join(sampled_rules)}"
        print(f"[Episode {ep + 1}/{episodes}] return={ret:.3f} outcome={outcome}{rule_summary}")

        if outcome == "aborted":
            print("[Info] Evaluation stopped by user.")
            break
        ep += 1

    if visualize and HAS_PYGAME and screen_bundle and auto_quit:
        pygame.quit()

    avg_ret = float(np.mean(returns)) if returns else 0.0
    print(f"[Summary] episodes={len(returns)} success={successes} ({100.0 * successes / max(1, len(returns)):.1f}%) avg_return={avg_ret:.3f}")
    return returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate shared-policy SAC on Majestro NavMesh.")
    ap.add_argument("--actor-path", type=str, default=DEFAULT_ACTOR_PATH)
    ap.add_argument("--actor-source", type=str, default=DEFAULT_ACTOR_SOURCE, choices=["latest", "single-best"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--no-visualize", action="store_true", default=False)
    ap.add_argument("--move-step-size", type=float, default=120.0)
    ap.add_argument("--tactical-target-radius", type=float, default=600.0)
    ap.add_argument("--num-other-agents", type=int, default=4)
    ap.add_argument("--observed-other-agents", type=int, default=3)
    ap.add_argument("--agent-radius", type=float, default=90.0)
    ap.add_argument("--sense-radius", type=float, default=1000.0)
    ap.add_argument("--goal-spawn-min-scale", type=float, default=4.0)
    ap.add_argument("--agent-spawn-min-scale", type=float, default=2.0)
    ap.add_argument("--agent-spawn-max-scale", type=float, default=3.0)
    ap.add_argument("--moving-goal", action="store_true", default=True)
    ap.add_argument("--moving-goal-speed-scale", type=float, default=(1.0 / 3.0))
    ap.add_argument("--spawn-agents-near-goal", action="store_true", default=True)
    ap.add_argument("--random-detour-mode", action="store_true", default=False,
                    help="Ignore policy actors and repeatedly move toward random targets sampled inside sense radius via Detour.")
    ap.add_argument("--role-rule", type=str, default="fixed", choices=["fixed", "melee_dps", "ranged_dps"])
    ap.add_argument("--agent-role-rules", type=str, default="melee_dps,melee_dps,melee_dps,ranged_dps,ranged_dps",
                    help="Comma-separated per-agent heuristic list. Length must be 1 or num_agents. Example: fixed,melee_dps,ranged_dps,fixed,fixed")
    ap.add_argument(
        "--agent-role-rule-pool",
        type=str,
        default="melee_dps,melee_dps,melee_dps,ranged_dps,ranged_dps;rdps,rdps,rdps;mdps,mdps,mdps,mdps;mdps;rdps,rdps,mdps,mdps,mdps",
        help="Semicolon-separated pool of per-episode heuristic sets. Each set is comma-separated. Shorter sets are expanded by repeating the last rule.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path = args.actor_path
    num_agents = 1 + int(args.num_other_agents)
    agent_role_rules = parse_agent_role_rules(args.agent_role_rules, num_agents)
    agent_role_rule_pool = parse_agent_role_rule_pool(args.agent_role_rule_pool, num_agents)

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
        moving_goal_enabled=bool(args.moving_goal),
        moving_goal_speed_scale=args.moving_goal_speed_scale,
        spawn_agents_near_goal=bool(args.spawn_agents_near_goal),
        role_rule=args.role_rule,
        agent_role_rules=agent_role_rules,
    )
    if agent_role_rule_pool is not None:
        env.agent_role_rule_pool = [list(rules) for rules in agent_role_rule_pool]
        pool_summary = " ; ".join(",".join(rules) for rules in agent_role_rule_pool)
        print(f"[ROLE-POOL] {len(agent_role_rule_pool)} sets | {pool_summary}")

    if not args.random_detour_mode:
        if args.actor_source == "latest" and (not os.path.exists(actor_path)):
            print(f"[WARN] {actor_path} not found. Train with Test.py first.")
            sys.exit(0)

    obs_dim = int(getattr(env, "single_agent_obs_dim", env.observation_space.shape[-1]))
    act_dim = int(getattr(env, "single_agent_act_dim", env.action_space.shape[-1]))
    role_bundles = {}
    if not args.random_detour_mode:
        actor_states = load_actor_state_map(args.actor_source, actor_path, POLICY_ROLE_IDS)
        for role_id in POLICY_ROLE_IDS:
            actor = GaussianPolicy(obs_dim, act_dim).to(device)
            state_dict = actor_states.get(role_name(role_id))
            if state_dict is None:
                raise RuntimeError(f"Checkpoint is missing actor for role '{role_name(role_id)}'.")
            actor.load_state_dict(state_dict)
            actor.eval()
            role_bundles[int(role_id)] = {"actor": actor}
    else:
        print("[MODE] random-detour-mode enabled")

    run_multiple_evaluations(
        env,
        role_bundles,
        episodes=args.episodes,
        scale=args.scale,
        visualize=(HAS_PYGAME and (not args.no_visualize)),
        visualize_every=1,
        auto_quit=True,
        save_last_csv=str(Path("last_eval_traj.csv").resolve()),
        random_detour_mode=bool(args.random_detour_mode),
    )
