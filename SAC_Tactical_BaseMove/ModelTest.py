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

DETOUR_PATH_COLOR = (150, 150, 150)
BUTTON_BG = (44, 52, 60)
BUTTON_ACTIVE = (74, 112, 74)
BUTTON_TEXT = (235, 235, 235)
BUTTON_BORDER = (110, 120, 130)


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


def _apply_agent_role_rules(env, chosen):
    if chosen is None:
        return
    chosen = [str(x).strip().lower() for x in chosen]
    if hasattr(env, "configure_agent_group") and callable(env.configure_agent_group):
        env.configure_agent_group(chosen)
    else:
        env.agent_role_rules = list(chosen)
    setattr(env, "_current_agent_role_rule_sample", list(chosen))
    setattr(env, "_current_agent_role_rule_sample_index", None)


def make_world_to_screen(bounds_min, bounds_max, scale, y_scale=1.0, min_width=640, min_height=720):
    min_x, min_z = float(bounds_min[0]), float(bounds_min[1])
    max_x, max_z = float(bounds_max[0]), float(bounds_max[1])
    render_width = max(1, int((max_x - min_x) * scale))
    render_height = max(1, int((max_z - min_z) * scale * y_scale))
    width = max(int(min_width), render_width, 1)
    height = max(int(min_height), render_height, 1)
    pad_x = max(0, (width - render_width) // 2)
    pad_y = max(0, (height - render_height) // 2)

    def world_to_screen(p):
        x, z = float(p[0]), float(p[1])
        sx = pad_x + int((x - min_x) * scale)
        sy = pad_y + int((max_z - z) * scale * y_scale)
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


def _snapshot_env_sim_state(env):
    return {
        "agent_positions": np.asarray(env.agent_positions, dtype=np.float32).copy(),
        "agent_heights": np.asarray(env.agent_heights, dtype=np.float32).copy(),
        "agent_velocities": np.asarray(env.agent_velocities, dtype=np.float32).copy(),
        "current_detour_waypoints": np.asarray(env.current_detour_waypoints, dtype=np.float32).copy(),
        "current_detour_waypoint_heights": np.asarray(env.current_detour_waypoint_heights, dtype=np.float32).copy(),
        "arrived_agents": np.asarray(getattr(env, "_arrived_agents", np.zeros((len(env.agent_positions),), dtype=bool)), dtype=bool).copy(),
        "agent_pos": np.asarray(env.agent_pos, dtype=np.float32).copy(),
        "agent_height": float(env.agent_height),
    }


def _apply_env_sim_state(env, state):
    env.agent_positions = np.asarray(state["agent_positions"], dtype=np.float32).copy()
    env.agent_heights = np.asarray(state["agent_heights"], dtype=np.float32).copy()
    env.agent_velocities = np.asarray(state["agent_velocities"], dtype=np.float32).copy()
    env.current_detour_waypoints = np.asarray(state["current_detour_waypoints"], dtype=np.float32).copy()
    env.current_detour_waypoint_heights = np.asarray(state["current_detour_waypoint_heights"], dtype=np.float32).copy()
    env._arrived_agents = np.asarray(state["arrived_agents"], dtype=bool).copy()
    env.agent_pos = np.asarray(state["agent_pos"], dtype=np.float32).copy()
    env.agent_height = float(state["agent_height"])


def _copy_env_runtime_state(src_env, dst_env):
    state = _snapshot_env_sim_state(src_env)
    _apply_env_sim_state(dst_env, state)
    attrs = [
        "goal_pos",
        "goal_height",
        "steps",
        "max_steps",
        "agent_role_ids",
        "role_targets",
        "last_target_offsets",
        "_prev_geo",
        "_prev_success_mask",
        "_prev_in_sense_mask",
        "_stall_best",
        "_stall_wait",
        "_episode_success_rewarded",
        "bounds_min",
        "bounds_max",
    ]
    for name in attrs:
        if hasattr(src_env, name):
            value = getattr(src_env, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            elif isinstance(value, list):
                value = list(value)
            setattr(dst_env, name, value)


def _init_detour_reference_state(env):
    return {
        "agent_positions": np.asarray(env.agent_positions, dtype=np.float32).copy(),
        "agent_heights": np.asarray(env.agent_heights, dtype=np.float32).copy(),
        "agent_velocities": np.zeros_like(np.asarray(env.agent_velocities, dtype=np.float32)),
        "current_detour_waypoints": np.full_like(np.asarray(env.current_detour_waypoints, dtype=np.float32), np.nan),
        "current_detour_waypoint_heights": np.full_like(np.asarray(env.current_detour_waypoint_heights, dtype=np.float32), np.nan),
        "arrived_agents": (np.linalg.norm(np.asarray(env.goal_pos, dtype=np.float32)[None, :] - np.asarray(env.agent_positions, dtype=np.float32), axis=1) <= float(getattr(env, "success_radius", 0.0))),
        "agent_pos": np.asarray(env.agent_pos, dtype=np.float32).copy(),
        "agent_height": float(env.agent_height),
        "collision_total": 0,
    }


def _compute_detour_preview_paths(env):
    preview_paths = []
    positions = np.asarray(env.agent_positions, dtype=np.float32)
    heights = np.asarray(env.agent_heights, dtype=np.float32)
    for idx, pos in enumerate(positions):
        start_height = float(heights[idx]) if idx < len(heights) else None
        path = recover_descent_path_world(env, pos, start_height=start_height, max_len=256)
        preview_paths.append([np.asarray(p, dtype=np.float32).copy() for p in path])
    return preview_paths


def _step_detour_reference(env, ref_state):
    working = {
        "agent_positions": np.asarray(ref_state["agent_positions"], dtype=np.float32).copy(),
        "agent_heights": np.asarray(ref_state["agent_heights"], dtype=np.float32).copy(),
        "agent_velocities": np.asarray(ref_state["agent_velocities"], dtype=np.float32).copy(),
        "current_detour_waypoints": np.asarray(ref_state["current_detour_waypoints"], dtype=np.float32).copy(),
        "current_detour_waypoint_heights": np.asarray(ref_state["current_detour_waypoint_heights"], dtype=np.float32).copy(),
        "arrived_agents": np.asarray(ref_state["arrived_agents"], dtype=bool).copy(),
        "agent_pos": np.asarray(ref_state["agent_pos"], dtype=np.float32).copy(),
        "agent_height": float(ref_state["agent_height"]),
    }
    _apply_env_sim_state(env, working)

    old_positions = env.agent_positions.copy()
    collisions = np.zeros((env.num_agents,), dtype=bool)
    collision_detected = np.zeros((env.num_agents,), dtype=bool)
    tactical_targets = np.zeros_like(env.agent_positions)
    goal_targets = np.repeat(np.asarray(env.goal_pos, dtype=np.float32).reshape(1, 2), env.num_agents, axis=0)

    for idx in range(env.num_agents):
        old_pos = old_positions[idx]
        if bool(env._arrived_agents[idx]):
            env.agent_positions[idx] = old_pos
            env.agent_velocities[idx] = np.zeros((2,), dtype=np.float32)
            tactical_targets[idx] = np.asarray(env.goal_pos, dtype=np.float32)
            collision_detected[idx] = False
            continue
        waypoint = None
        waypoint_height = float("nan")
        if getattr(env, "_detour_enabled", False):
            reach_tol = max(env._grid_cell_size * 0.50, env.step_size * 0.35)
            cached_waypoint = env.current_detour_waypoints[idx]
            cached_waypoint_height = float(env.current_detour_waypoint_heights[idx])
            if np.all(np.isfinite(cached_waypoint)):
                cached_dist = float(np.linalg.norm(cached_waypoint - old_pos))
                if cached_dist > reach_tol and np.isfinite(cached_waypoint_height):
                    waypoint = cached_waypoint.astype(np.float32)
                    waypoint_height = cached_waypoint_height
            if waypoint is None:
                detour_result = env._detour_next_waypoint_with_min_progress(
                    old_pos,
                    np.asarray(env.goal_pos, dtype=np.float32),
                    min_progress=max(env._grid_cell_size * 0.50, env.step_size * 0.25),
                    height=float(env.agent_heights[idx]),
                    target_height=float(env.goal_height),
                )
                if detour_result is not None:
                    waypoint, waypoint_height = detour_result
                    env.current_detour_waypoints[idx] = waypoint.astype(np.float32)
                    env.current_detour_waypoint_heights[idx] = float(waypoint_height)
                else:
                    env.current_detour_waypoints[idx] = np.array([np.nan, np.nan], dtype=np.float32)
                    env.current_detour_waypoint_heights[idx] = np.float32(np.nan)
        else:
            waypoint = env._geo_next_waypoint(old_pos, max_search=3)

        tactical_target = np.asarray(env.goal_pos, dtype=np.float32) if waypoint is None else np.asarray(waypoint, dtype=np.float32)
        to_target = tactical_target - old_pos
        target_dist = float(np.linalg.norm(to_target))
        if target_dist > env.step_size and target_dist > 1e-6:
            movement_target = old_pos + (to_target / target_dist) * env.step_size
        else:
            movement_target = tactical_target

        detour_priority_allowed = waypoint is not None and np.isfinite(waypoint_height)
        if detour_priority_allowed:
            move_ratio = 1.0 if target_dist <= 1e-6 else min(1.0, env.step_size / max(target_dist, 1e-6))
            detour_height = float(env.agent_heights[idx]) + (float(waypoint_height) - float(env.agent_heights[idx])) * move_ratio
            if np.isfinite(detour_height) and not env._collides_with_other_agents(movement_target, ignore_index=idx):
                new_pos = movement_target.astype(np.float32)
                new_height = float(detour_height)
                collided = False
            else:
                new_pos, new_height, collided = env._move_with_agent_avoidance(
                    old_pos,
                    movement_target,
                    ignore_index=idx,
                    start_height=float(env.agent_heights[idx]),
                )
        else:
            new_pos, new_height, collided = env._move_with_agent_avoidance(
                old_pos,
                movement_target,
                ignore_index=idx,
                start_height=float(env.agent_heights[idx]),
            )

        env.agent_positions[idx] = new_pos
        env.agent_heights[idx] = new_height
        env.agent_velocities[idx] = new_pos - old_pos
        tactical_targets[idx] = tactical_target
        collisions[idx] = collided
        collision_detected[idx] = env._collides_with_other_agents(new_pos, ignore_index=idx)
        if float(np.linalg.norm(np.asarray(env.goal_pos, dtype=np.float32) - new_pos)) <= float(getattr(env, "success_radius", 0.0)):
            env._arrived_agents[idx] = True
            collisions[idx] = False
            collision_detected[idx] = False

        current_wp = env.current_detour_waypoints[idx]
        if np.all(np.isfinite(current_wp)):
            if float(np.linalg.norm(current_wp - new_pos)) <= max(env._grid_cell_size * 0.50, env.step_size * 0.20):
                env.current_detour_waypoints[idx] = np.array([np.nan, np.nan], dtype=np.float32)
                env.current_detour_waypoint_heights[idx] = np.float32(np.nan)

    env.agent_pos = env.agent_positions[0].copy()
    env.agent_height = float(env.agent_heights[0])

    dists = np.linalg.norm(np.asarray(env.goal_pos, dtype=np.float32)[None, :] - env.agent_positions, axis=1).astype(np.float32)
    in_sense_mask = dists <= float(env.sense_radius)
    ref_state["agent_positions"] = env.agent_positions.copy()
    ref_state["agent_heights"] = env.agent_heights.copy()
    ref_state["agent_velocities"] = env.agent_velocities.copy()
    ref_state["current_detour_waypoints"] = env.current_detour_waypoints.copy()
    ref_state["current_detour_waypoint_heights"] = env.current_detour_waypoint_heights.copy()
    ref_state["arrived_agents"] = np.asarray(env._arrived_agents, dtype=bool).copy()
    ref_state["agent_pos"] = env.agent_pos.copy()
    ref_state["agent_height"] = float(env.agent_height)
    ref_state["collision_total"] = int(ref_state.get("collision_total", 0)) + int(np.count_nonzero(collision_detected))

    info = {
        "agent_positions": env.agent_positions.copy(),
        "agent_heights": env.agent_heights.copy(),
        "agent_velocities": env.agent_velocities.copy(),
        "collided": collision_detected.copy(),
        "collision_handled": collisions.copy(),
        "collision_total": int(ref_state["collision_total"]),
        "goal_pos": np.asarray(env.goal_pos, dtype=np.float32).copy(),
        "in_sense_mask": in_sense_mask.copy(),
        "role_ids": np.full((env.num_agents,), 2, dtype=np.int32),
        "tactical_target": tactical_targets.copy(),
        "role_targets": goal_targets.copy(),
    }
    return info


def _draw_sim_panel(screen, env, world_to_screen, x_offset, font, title, trajs, sim_info, step, max_steps, extra_lines, scale):
    def panel_world_to_screen(p):
        sx, sy = world_to_screen(p)
        return sx + x_offset, sy

    draw_navmesh_overlay(screen, env, panel_world_to_screen)

    sense_px = max(1, int(round(float(getattr(env, "sense_radius", 0.0)) * float(scale))))

    positions = np.asarray(sim_info.get("agent_positions", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    collided = np.asarray(sim_info.get("collided", np.zeros((len(positions),), dtype=bool)), dtype=bool).reshape(-1)
    role_ids = np.asarray(sim_info.get("role_ids", np.zeros((len(trajs),), dtype=np.int32)), dtype=np.int32).reshape(-1)
    tactical_target = np.asarray(sim_info.get("tactical_target", np.zeros_like(positions)), dtype=np.float32)
    role_targets = np.asarray(sim_info.get("role_targets", np.zeros_like(positions)), dtype=np.float32)
    preview_paths = sim_info.get("preview_paths") or []

    if len(positions) > 0:
        for idx, pos in enumerate(positions):
            role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
            color = ROLE_COLORS.get(role_id, (160, 160, 160))
            pygame.draw.circle(screen, color, panel_world_to_screen(pos), sense_px, 1)

    for path in preview_paths:
        if len(path) < 2:
            continue
        pygame.draw.lines(screen, DETOUR_PATH_COLOR, False, [panel_world_to_screen(p) for p in path], 3)

    for idx, points in enumerate(trajs):
        if len(points) < 2:
            continue
        role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
        color = ROLE_COLORS.get(role_id, (160, 160, 160))
        pygame.draw.lines(screen, color, False, [panel_world_to_screen(p) for p in points], 2)

    if tactical_target.size > 0:
        for idx, target in enumerate(np.asarray(tactical_target, dtype=np.float32)):
            color = (120, 120, 240) if idx == 0 else (110, 110, 170)
            pygame.draw.circle(screen, color, panel_world_to_screen(target), 4 if idx == 0 else 3)

    if role_targets.size > 0:
        for idx, role_target in enumerate(np.asarray(role_targets, dtype=np.float32)):
            role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
            color = ROLE_COLORS.get(role_id, (170, 110, 110))
            pygame.draw.circle(screen, color, panel_world_to_screen(role_target), 3, 1)

    for idx, other in enumerate(positions):
        role_id = int(role_ids[idx]) if idx < len(role_ids) else 0
        color = ROLE_COLORS.get(role_id, (200, 140, 70))
        if idx < len(collided) and bool(collided[idx]):
            color = (235, 70, 70)
        sx, sy = panel_world_to_screen(other)
        pygame.draw.circle(screen, color, (sx, sy), 4)
        surf = font.render(f"A{idx}", True, color)
        screen.blit(surf, (sx + 6, sy - 10))

    goal_pos = np.asarray(sim_info.get("goal_pos", np.asarray(env.goal_pos, dtype=np.float32)), dtype=np.float32)
    if goal_pos.size >= 2:
        pygame.draw.circle(screen, (230, 90, 90), panel_world_to_screen(goal_pos), 6)
    if len(positions) > 0:
        pygame.draw.circle(screen, (255, 255, 255), panel_world_to_screen(positions[0]), 5, 1)

    pygame.draw.line(screen, (55, 60, 66), (x_offset, 0), (x_offset, screen.get_height()), 1)
    title_surf = font.render(title, True, (255, 220, 140))
    screen.blit(title_surf, (x_offset + 8, 8))

    y = 28
    for line in [f"Step: {step + 1}/{max_steps}", *extra_lines]:
        surf = font.render(line, True, (220, 220, 220))
        screen.blit(surf, (x_offset + 8, y))
        y += 18


def _relative_rate_vs_detour(actor_count: int, detour_count: int) -> float:
    if detour_count > 0:
        return 100.0 * float(actor_count) / float(detour_count)
    return 100.0 if int(actor_count) == 0 else 0.0


def _build_control_buttons(screen_width: int):
    labels = ["Pause", "Fast"]
    buttons = []
    x = screen_width - 196
    y = 8
    w = 92
    h = 28
    gap = 8
    for idx, label in enumerate(labels):
        buttons.append({"label": label, "rect": pygame.Rect(x + idx * (w + gap), y, w, h)})
    return buttons


def _draw_control_buttons(screen, font, buttons, paused, fast_mode):
    states = {
        "Pause": "Paused" if paused else "Pause",
        "Fast": "On" if fast_mode else "Fast",
    }
    for button in buttons:
        label = button["label"]
        rect = button["rect"]
        active = (label == "Pause" and paused) or (label == "Fast" and fast_mode)
        pygame.draw.rect(screen, BUTTON_ACTIVE if active else BUTTON_BG, rect, border_radius=6)
        pygame.draw.rect(screen, BUTTON_BORDER, rect, width=1, border_radius=6)
        surf = font.render(f"{label}:{states[label]}", True, BUTTON_TEXT)
        text_rect = surf.get_rect(center=rect.center)
        screen.blit(surf, text_rect)


def evaluate_once(env, actor, max_steps=None, scale=0.03, screen_bundle=None, visualize=True, save_csv_path=None, visualize_detour_compare=True, detour_env=None):
    obs, info = reset_env_compat(env)
    if visualize_detour_compare and detour_env is not None:
        reset_env_compat(detour_env)
        _copy_env_runtime_state(env, detour_env)
    actor.eval()

    start_pos = np.array(env.agent_pos, dtype=np.float32).copy()
    goal_pos = np.array(env.goal_pos, dtype=np.float32).copy()
    max_steps = int(max_steps or getattr(env, "max_steps", 300))
    initial_dists = np.linalg.norm(
        np.asarray(env.goal_pos, dtype=np.float32)[None, :] - np.asarray(env.agent_positions, dtype=np.float32),
        axis=1,
    )
    initial_total_agents = int(len(initial_dists))
    initial_in_sense = int(np.count_nonzero(initial_dists <= float(getattr(env, "sense_radius", 0.0))))
    farthest_agent_idx = int(np.argmax(initial_dists)) if initial_total_agents > 0 else -1
    farthest_agent_dist = float(initial_dists[farthest_agent_idx]) if initial_total_agents > 0 else 0.0
    farthest_agent_geo_dist = float("nan")
    geo_distance_fn = getattr(env, "_geo_distance", None)
    if initial_total_agents > 0 and callable(geo_distance_fn):
        try:
            geo_val = geo_distance_fn(np.asarray(env.agent_positions, dtype=np.float32)[farthest_agent_idx])
            if geo_val is not None:
                farthest_agent_geo_dist = float(geo_val)
        except Exception:
            farthest_agent_geo_dist = float("nan")

    agent_trajs = [[np.array(pos, dtype=np.float32).copy()] for pos in np.asarray(env.agent_positions, dtype=np.float32)]
    detour_ref_state = _init_detour_reference_state(detour_env) if (visualize_detour_compare and detour_env is not None) else None
    detour_trajs = [[np.array(pos, dtype=np.float32).copy()] for pos in np.asarray(detour_env.agent_positions if detour_env is not None else env.agent_positions, dtype=np.float32)]
    detour_preview_paths = _compute_detour_preview_paths(detour_env if detour_env is not None else env)
    detour_info = {
        "agent_positions": np.asarray(env.agent_positions, dtype=np.float32).copy(),
        "goal_pos": np.asarray(env.goal_pos, dtype=np.float32).copy(),
        "in_sense_mask": initial_dists <= float(getattr(env, "sense_radius", 0.0)),
        "collided": np.zeros((initial_total_agents,), dtype=bool),
        "collision_total": 0,
        "role_ids": np.full((initial_total_agents,), 2, dtype=np.int32),
        "tactical_target": np.repeat(np.asarray(env.goal_pos, dtype=np.float32).reshape(1, 2), initial_total_agents, axis=0),
        "role_targets": np.repeat(np.asarray(env.goal_pos, dtype=np.float32).reshape(1, 2), initial_total_agents, axis=0),
        "preview_paths": detour_preview_paths,
    }
    traj = [start_pos.copy()]
    screen = clock = font = None
    world_to_screen = None

    if visualize and HAS_PYGAME:
        if screen_bundle is None:
            pygame.init()
            width, height, world_to_screen = make_world_to_screen(env.bounds_min, env.bounds_max, scale, y_scale=0.90)
            compare_enabled = bool(visualize_detour_compare)
            screen = pygame.display.set_mode((width * (2 if compare_enabled else 1), height))
            pygame.display.set_caption("ModelTest - Majestro NavMesh")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont("consolas", 16)
            screen_bundle = (screen, clock, font, world_to_screen, width, compare_enabled, False)
        else:
            if len(screen_bundle) >= 7:
                screen, clock, font, world_to_screen, width, compare_enabled, persisted_fast_mode = screen_bundle
            else:
                screen, clock, font, world_to_screen, width, compare_enabled = screen_bundle
                persisted_fast_mode = False

    ep_ret = 0.0
    total_collisions = 0
    final_info = {}
    env_terminated = False
    env_truncated = False
    user_aborted = False
    paused = False
    fast_mode = bool(persisted_fast_mode) if 'persisted_fast_mode' in locals() else False
    controls = _build_control_buttons(screen.get_width()) if (visualize and HAS_PYGAME and screen is not None) else []

    for step in range(max_steps):
        draw_enabled = visualize and (not fast_mode)
        if visualize and HAS_PYGAME and screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    user_aborted = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    user_aborted = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in controls:
                        if button["rect"].collidepoint(event.pos):
                            label = button["label"]
                            if label == "Pause":
                                paused = not paused
                            elif label == "Fast":
                                fast_mode = not fast_mode
                                if fast_mode:
                                    paused = False

        if user_aborted:
            break

        if paused:
            if draw_enabled and HAS_PYGAME and screen is not None:
                _draw_control_buttons(screen, font, controls, paused, fast_mode)
                pygame.display.flip()
                clock.tick(30)
            continue

        action = policy_act(actor, obs)
        obs, reward, env_terminated, env_truncated, final_info = env.step(action)
        ep_ret += float(np.mean(np.asarray(reward, dtype=np.float32)))
        collided = np.asarray(final_info.get("collided", np.zeros((initial_total_agents,), dtype=bool)), dtype=bool).reshape(-1)
        total_collisions += int(np.count_nonzero(collided))
        if visualize_detour_compare and detour_env is not None and detour_ref_state is not None:
            detour_info = _step_detour_reference(detour_env, detour_ref_state)
            detour_info["preview_paths"] = detour_preview_paths
            for idx, pos in enumerate(np.asarray(detour_info.get("agent_positions", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)):
                if idx < len(detour_trajs):
                    detour_trajs[idx].append(pos.copy())
        traj.append(np.array(env.agent_pos, dtype=np.float32).copy())
        for idx, pos in enumerate(np.asarray(env.agent_positions, dtype=np.float32)):
            if idx < len(agent_trajs):
                agent_trajs[idx].append(pos.copy())

        if draw_enabled and HAS_PYGAME and screen is not None:
            screen.fill((14, 16, 20))
            role_ids = final_info.get("role_ids")
            if role_ids is None:
                role_ids = np.zeros((len(agent_trajs),), dtype=np.int32)
            else:
                role_ids = np.asarray(role_ids).reshape(-1)

            detour_entered = int(np.count_nonzero(np.asarray(detour_info.get('in_sense_mask', np.zeros((initial_total_agents,), dtype=bool)), dtype=bool).reshape(-1))) if visualize_detour_compare else 0
            vs_detour_rate = _relative_rate_vs_detour(
                int(np.count_nonzero(np.asarray(final_info.get('in_sense_mask', np.zeros((initial_total_agents,), dtype=bool)), dtype=bool).reshape(-1))),
                detour_entered,
            ) if visualize_detour_compare else 0.0
            actor_lines = [
                f"Return: {ep_ret:.3f}",
                f"Farthest: A{farthest_agent_idx}",
                f"Straight: {farthest_agent_dist:.1f}  Geo: {farthest_agent_geo_dist:.1f}",
                f"Entered: {int(np.count_nonzero(np.asarray(final_info.get('in_sense_mask', np.zeros((initial_total_agents,), dtype=bool)), dtype=bool).reshape(-1)))}/{initial_total_agents}",
                f"Collisions: {total_collisions}",
            ]
            if visualize_detour_compare:
                actor_lines.append(f"VsDetour: {vs_detour_rate:.1f}%")
            if role_ids is not None:
                role_ids_arr = np.asarray(role_ids, dtype=np.int32).reshape(-1)
                none_count = int(np.sum(role_ids_arr == -1))
                base_move_count = int(np.sum(role_ids_arr == 2))
                actor_lines.append(f"Roles: none={none_count} base_move={base_move_count}")

            actor_panel_info = {
                "agent_positions": np.asarray(final_info.get("agent_positions", env.agent_positions), dtype=np.float32),
                "goal_pos": np.asarray(goal_pos, dtype=np.float32),
                "in_sense_mask": np.asarray(final_info.get("in_sense_mask", np.zeros((initial_total_agents,), dtype=bool)), dtype=bool),
                "collided": np.asarray(final_info.get("collided", np.zeros((initial_total_agents,), dtype=bool)), dtype=bool),
                "collision_total": total_collisions,
                "role_ids": role_ids.copy(),
                "tactical_target": np.asarray(final_info.get("tactical_target", np.asarray(goal_pos, dtype=np.float32)), dtype=np.float32),
                "role_targets": np.asarray(final_info.get("role_targets", np.repeat(np.asarray(goal_pos, dtype=np.float32).reshape(1, 2), initial_total_agents, axis=0)), dtype=np.float32),
            }
            _draw_sim_panel(
                screen,
                env,
                world_to_screen,
                width if visualize_detour_compare else 0,
                font,
                "BaseMove Actor",
                agent_trajs,
                actor_panel_info,
                step,
                max_steps,
                actor_lines,
                scale,
            )

            if visualize_detour_compare:
                detour_lines = [
                    f"Entered: {detour_entered}/{initial_total_agents}",
                    f"Collisions: {int(detour_info.get('collision_total', 0))}",
                    f"Agents: {len(detour_trajs)}",
                ]
                _draw_sim_panel(
                    screen,
                    env,
                    world_to_screen,
                    0,
                    font,
                    "Detour Only",
                    detour_trajs,
                    detour_info,
                    step,
                    max_steps,
                    detour_lines,
                    scale,
                )

            _draw_control_buttons(screen, font, controls, paused, fast_mode)
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

    terminal_mask = np.asarray(final_info.get("in_sense_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
    terminal_total_agents = int(len(terminal_mask)) if len(terminal_mask) > 0 else initial_total_agents
    terminal_in_sense = int(np.count_nonzero(terminal_mask))
    terminal_rate = 100.0 * terminal_in_sense / max(1, terminal_total_agents)
    detour_terminal_mask = np.asarray(detour_info.get("in_sense_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1) if visualize_detour_compare else np.zeros((0,), dtype=bool)
    detour_terminal_in_sense = int(np.count_nonzero(detour_terminal_mask)) if len(detour_terminal_mask) > 0 else 0
    vs_detour_rate = _relative_rate_vs_detour(terminal_in_sense, detour_terminal_in_sense) if visualize_detour_compare else 0.0

    metrics = {
        "start_in_sense": initial_in_sense,
        "end_in_sense": terminal_in_sense,
        "detour_end_in_sense": detour_terminal_in_sense,
        "vs_detour_rate": vs_detour_rate,
        "total_agents": terminal_total_agents,
        "end_rate": terminal_rate,
        "collisions": total_collisions,
        "farthest_agent_idx": farthest_agent_idx,
        "farthest_agent_dist": farthest_agent_dist,
        "farthest_agent_geo_dist": farthest_agent_geo_dist,
    }
    if visualize and HAS_PYGAME and screen is not None:
        screen_bundle = (screen, clock, font, world_to_screen, width, compare_enabled, fast_mode)
    return ep_ret, success, outcome, screen_bundle, metrics


def run_multiple_evaluations(
    env,
    detour_env,
    actor,
    episodes=10,
    max_steps=None,
    scale=0.03,
    visualize=True,
    visualize_every=1,
    auto_quit=True,
    save_last_csv=None,
    visualize_detour_compare=True,
):
    returns = []
    successes = 0
    total_start_in_sense = 0
    total_end_in_sense = 0
    total_detour_end_in_sense = 0
    total_collisions = 0
    total_agents = 0
    screen_bundle = None

    for ep in range(episodes):
        sampled_rules = maybe_sample_agent_role_rules(env)
        _apply_agent_role_rules(detour_env, sampled_rules)
        vis = visualize and ((ep % visualize_every) == 0)
        save_csv = save_last_csv if ep == episodes - 1 else None
        ret, succ, outcome, screen_bundle, metrics = evaluate_once(
            env,
            actor,
            max_steps=max_steps,
            scale=scale,
            screen_bundle=screen_bundle if vis else None,
            visualize=vis,
            save_csv_path=save_csv,
            visualize_detour_compare=visualize_detour_compare,
            detour_env=detour_env,
        )
        returns.append(ret)
        successes += int(succ)
        total_start_in_sense += int(metrics["start_in_sense"])
        total_end_in_sense += int(metrics["end_in_sense"])
        total_detour_end_in_sense += int(metrics.get("detour_end_in_sense", 0))
        total_collisions += int(metrics["collisions"])
        total_agents += int(metrics["total_agents"])
        rule_summary = "" if sampled_rules is None else f" agents={len(sampled_rules)}"
        episode_line = (
            f"[Episode {ep + 1}/{episodes}] return={ret:.3f} outcome={outcome}"
            f" start_in_sense={int(metrics['start_in_sense'])}/{int(metrics['total_agents'])}"
            f" farthest=A{int(metrics['farthest_agent_idx'])}"
            f" straight={float(metrics['farthest_agent_dist']):.1f}"
            f" geo={float(metrics['farthest_agent_geo_dist']):.1f}"
            f" in_sense_end={int(metrics['end_in_sense'])}/{int(metrics['total_agents'])}"
        )
        if visualize_detour_compare:
            episode_line += (
                f" detour_end={int(metrics['detour_end_in_sense'])}/{int(metrics['total_agents'])}"
                f" vs_detour={float(metrics['vs_detour_rate']):.1f}%"
            )
        episode_line += f" collisions={int(metrics['collisions'])}"
        episode_line += f" ({float(metrics['end_rate']):.1f}%){rule_summary}"
        print(episode_line)

        if outcome == "aborted":
            print("[Info] Evaluation stopped by user.")
            break

    if visualize and HAS_PYGAME and screen_bundle and auto_quit:
        pygame.quit()

    avg_ret = float(np.mean(returns)) if returns else 0.0
    end_rate = 100.0 * total_end_in_sense / max(1, total_agents)
    vs_detour_rate = _relative_rate_vs_detour(total_end_in_sense, total_detour_end_in_sense) if visualize_detour_compare else 0.0
    summary_line = (
        f"[Summary] episodes={len(returns)} success={successes} ({100.0 * successes / max(1, len(returns)):.1f}%) "
        f"start_in_sense={total_start_in_sense}/{total_agents} "
        f"in_sense_end={total_end_in_sense}/{total_agents} ({end_rate:.1f}%) "
    )
    if visualize_detour_compare:
        summary_line += (
            f"detour_end={total_detour_end_in_sense}/{total_agents} "
            f"vs_detour={vs_detour_rate:.1f}% "
        )
    summary_line += f"collisions={total_collisions} "
    summary_line += f"avg_return={avg_ret:.3f}"
    print(summary_line)
    return returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate shared-policy SAC on Majestro NavMesh.")
    ap.add_argument("--actor-path", type=str, default="sac_actor_best.pth")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.03)
    ap.add_argument("--no-visualize", action="store_true", default=False)
    ap.add_argument("--no-draw", action="store_true", default=False)
    ap.add_argument("--no-detour-compare", action="store_true", default=False)
    ap.add_argument("--move-step-size", type=float, default=120.0)
    ap.add_argument("--tactical-target-radius", type=float, default=600.0)
    ap.add_argument("--num-other-agents", type=int, default=4)
    ap.add_argument("--random-agent-count-min", type=int, default=18)
    ap.add_argument("--random-agent-count-max", type=int, default=20)
    ap.add_argument("--observed-other-agents", type=int, default=3)
    ap.add_argument("--agent-radius", type=float, default=90.0)
    ap.add_argument("--sense-radius", type=float, default=1000.0)
    ap.add_argument("--resolve-agent-collisions", action="store_true", default=False)
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
        resolve_agent_collisions=args.resolve_agent_collisions,
        goal_spawn_min_scale=args.goal_spawn_min_scale,
        agent_spawn_min_scale=args.agent_spawn_min_scale,
        agent_spawn_max_scale=args.agent_spawn_max_scale,
        role_rule="base_move_only",
        agent_role_rules=agent_role_rules,
        dynamic_horizon_kappa=1.3,
    )
    env.random_agent_count_min = int(args.random_agent_count_min)
    env.random_agent_count_max = int(max(args.random_agent_count_min, args.random_agent_count_max))
    detour_env = build_env(
        seed=args.seed,
        move_step_size=args.move_step_size,
        tactical_target_radius=args.tactical_target_radius,
        num_other_agents=args.num_other_agents,
        observed_other_agents=args.observed_other_agents,
        agent_radius=args.agent_radius,
        sense_radius=args.sense_radius,
        resolve_agent_collisions=args.resolve_agent_collisions,
        goal_spawn_min_scale=args.goal_spawn_min_scale,
        agent_spawn_min_scale=args.agent_spawn_min_scale,
        agent_spawn_max_scale=args.agent_spawn_max_scale,
        role_rule="base_move_only",
        agent_role_rules=agent_role_rules,
        dynamic_horizon_kappa=1.3,
    )
    detour_env.random_agent_count_min = env.random_agent_count_min
    detour_env.random_agent_count_max = env.random_agent_count_max
    print(f"[AGENT-COUNT] random total agents per episode: {env.random_agent_count_min}..{env.random_agent_count_max}")
    print(f"[ENV] resolve_agent_collisions={bool(args.resolve_agent_collisions)}")

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
        detour_env,
        actor,
        episodes=args.episodes,
        scale=args.scale,
        visualize=(HAS_PYGAME and (not args.no_visualize) and (not args.no_draw)),
        visualize_every=1,
        auto_quit=True,
        save_last_csv=str(Path("last_eval_traj.csv").resolve()),
        visualize_detour_compare=(not args.no_detour_compare),
    )
