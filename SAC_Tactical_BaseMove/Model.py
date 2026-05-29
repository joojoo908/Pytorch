# Model.py
# Base-move-only SAC trainer.

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ROLE_BASE_MOVE = 2
ROLE_NONE = -1
ROLE_IDS = (ROLE_BASE_MOVE,)


def role_name(role_id: int) -> str:
    if int(role_id) == ROLE_BASE_MOVE:
        return "base_move"
    if int(role_id) == ROLE_NONE:
        return "none"
    return f"role_{int(role_id)}"


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, r, bool(terminated or truncated), info
    if isinstance(out, tuple) and len(out) == 4:
        return out
    raise RuntimeError("Unsupported env.step(...) return format")


def _is_multi_agent_obs(obs) -> bool:
    return np.asarray(obs).ndim >= 2


def infer_single_agent_obs_dim(env, probe_obs) -> int:
    if hasattr(env, "single_agent_obs_dim"):
        return int(env.single_agent_obs_dim)
    arr = np.asarray(probe_obs)
    if arr.ndim >= 2:
        return int(arr.shape[-1])
    if arr.ndim == 1:
        return int(arr.shape[0])
    return int(len(probe_obs))


def infer_single_agent_act_dim(env) -> int:
    if hasattr(env, "single_agent_act_dim"):
        return int(env.single_agent_act_dim)
    if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        shape = tuple(int(x) for x in env.action_space.shape)
        if len(shape) >= 2:
            return int(shape[-1])
        if len(shape) == 1:
            return int(shape[0])
    return 2


def extract_role_success(info: Dict[str, Any]) -> Dict[str, bool]:
    success = False
    if isinstance(info, dict):
        role_ids = info.get("role_ids")
        success_mask = info.get("success_mask")
        if role_ids is not None and success_mask is not None:
            try:
                role_ids = np.asarray(role_ids, dtype=np.int32).reshape(-1)
                success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
                success = bool(np.any((role_ids == ROLE_BASE_MOVE) & success_mask))
            except Exception:
                success = False
    return {"base_move": success}


def is_diverse_tactical_success(info: Dict[str, Any]) -> bool:
    return bool(extract_role_success(info)["base_move"])


def maybe_sample_agent_role_rules(env):
    min_agents = int(getattr(env, "random_agent_count_min", 0) or 0)
    max_agents = int(getattr(env, "random_agent_count_max", 0) or 0)
    if max_agents >= max(1, min_agents):
        count = int(np.random.randint(max(1, min_agents), max_agents + 1))
    else:
        count = int(getattr(env, "num_agents", 1))
    chosen = ["base_move_only"] * count
    if hasattr(env, "configure_agent_group") and callable(env.configure_agent_group):
        env.configure_agent_group(chosen)
    else:
        env.agent_role_rules = list(chosen)
    setattr(env, "_current_agent_role_rule_sample", list(chosen))
    setattr(env, "_current_agent_role_rule_sample_index", None)
    return chosen


def get_env_role_ids(env, count: int) -> np.ndarray:
    return np.full((count,), ROLE_BASE_MOVE, dtype=np.int32)


def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def soft_update_(src: nn.Module, dst: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def _relative_rate_vs_detour(actor_count: int, detour_count: int) -> float:
    if int(detour_count) > 0:
        return 100.0 * float(actor_count) / float(detour_count)
    return 100.0 if int(actor_count) == 0 else 0.0


def _snapshot_env_runtime_state(env) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    attrs = [
        "agent_positions",
        "agent_heights",
        "agent_velocities",
        "current_detour_waypoints",
        "current_detour_waypoint_heights",
        "_arrived_agents",
        "agent_pos",
        "agent_height",
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
    ]
    for name in attrs:
        if hasattr(env, name):
            value = getattr(env, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            elif isinstance(value, list):
                value = list(value)
            state[name] = value
    return state


def _apply_env_runtime_state(env, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
        elif isinstance(value, list):
            value = list(value)
        setattr(env, name, value)


def _init_detour_reference_state(env) -> Dict[str, Any]:
    return {
        "agent_positions": np.asarray(env.agent_positions, dtype=np.float32).copy(),
        "agent_heights": np.asarray(env.agent_heights, dtype=np.float32).copy(),
        "agent_velocities": np.zeros_like(np.asarray(env.agent_velocities, dtype=np.float32)),
        "current_detour_waypoints": np.full_like(np.asarray(env.current_detour_waypoints, dtype=np.float32), np.nan),
        "current_detour_waypoint_heights": np.full_like(np.asarray(env.current_detour_waypoint_heights, dtype=np.float32), np.nan),
        "arrived_agents": np.asarray(getattr(env, "_arrived_agents", np.zeros((len(env.agent_positions),), dtype=bool)), dtype=bool).copy(),
        "agent_pos": np.asarray(env.agent_pos, dtype=np.float32).copy(),
        "agent_height": float(env.agent_height),
    }


def _step_detour_reference(env, ref_state: Dict[str, Any]) -> Dict[str, Any]:
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
    env.agent_positions = working["agent_positions"]
    env.agent_heights = working["agent_heights"]
    env.agent_velocities = working["agent_velocities"]
    env.current_detour_waypoints = working["current_detour_waypoints"]
    env.current_detour_waypoint_heights = working["current_detour_waypoint_heights"]
    env._arrived_agents = working["arrived_agents"]
    env.agent_pos = working["agent_pos"]
    env.agent_height = working["agent_height"]

    old_positions = env.agent_positions.copy()
    collisions = np.zeros((env.num_agents,), dtype=bool)

    for idx in range(env.num_agents):
        old_pos = old_positions[idx]
        if bool(env._arrived_agents[idx]):
            env.agent_positions[idx] = old_pos
            env.agent_velocities[idx] = np.zeros((2,), dtype=np.float32)
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
        collisions[idx] = collided

        if float(np.linalg.norm(np.asarray(env.goal_pos, dtype=np.float32) - new_pos)) <= float(getattr(env, "success_radius", 0.0)):
            env._arrived_agents[idx] = True
            collisions[idx] = False

        current_wp = env.current_detour_waypoints[idx]
        if np.all(np.isfinite(current_wp)):
            if float(np.linalg.norm(current_wp - new_pos)) <= max(env._grid_cell_size * 0.50, env.step_size * 0.20):
                env.current_detour_waypoints[idx] = np.array([np.nan, np.nan], dtype=np.float32)
                env.current_detour_waypoint_heights[idx] = np.float32(np.nan)

    env.agent_pos = env.agent_positions[0].copy()
    env.agent_height = float(env.agent_heights[0])
    ref_state["agent_positions"] = env.agent_positions.copy()
    ref_state["agent_heights"] = env.agent_heights.copy()
    ref_state["agent_velocities"] = env.agent_velocities.copy()
    ref_state["current_detour_waypoints"] = env.current_detour_waypoints.copy()
    ref_state["current_detour_waypoint_heights"] = env.current_detour_waypoint_heights.copy()
    ref_state["arrived_agents"] = np.asarray(env._arrived_agents, dtype=bool).copy()
    ref_state["agent_pos"] = env.agent_pos.copy()
    ref_state["agent_height"] = float(env.agent_height)
    dists = np.linalg.norm(np.asarray(env.goal_pos, dtype=np.float32)[None, :] - env.agent_positions, axis=1).astype(np.float32)
    return {
        "in_sense_mask": dists <= float(env.sense_radius),
        "collision_handled": collisions.copy(),
    }


def _run_detour_baseline_terminal_count(env, initial_state: Dict[str, Any], horizon: int) -> int:
    _apply_env_runtime_state(env, initial_state)
    ref_state = _init_detour_reference_state(env)
    info: Dict[str, Any] = {"in_sense_mask": np.zeros((env.num_agents,), dtype=bool)}
    for _ in range(max(0, int(horizon))):
        info = _step_detour_reference(env, ref_state)
    terminal_mask = np.asarray(info.get("in_sense_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
    return int(np.count_nonzero(terminal_mask))


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, obs_dtype=np.float32, act_dtype=np.float32):
        self.capacity = int(capacity)
        self.obs = deque(maxlen=self.capacity)
        self.act = deque(maxlen=self.capacity)
        self.rew = deque(maxlen=self.capacity)
        self.nobs = deque(maxlen=self.capacity)
        self.done = deque(maxlen=self.capacity)
        self._obs_dtype = obs_dtype
        self._act_dtype = act_dtype

    def __len__(self):
        return len(self.obs)

    def push(self, s, a, r, ns, d):
        self.obs.append(np.asarray(s, dtype=self._obs_dtype))
        self.act.append(np.asarray(a, dtype=self._act_dtype))
        self.rew.append(np.asarray(r, dtype=np.float32))
        self.nobs.append(np.asarray(ns, dtype=self._obs_dtype))
        self.done.append(np.asarray(d, dtype=np.float32))

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.obs), size=batch_size)
        return (
            np.stack([self.obs[i] for i in idx], axis=0),
            np.stack([self.act[i] for i in idx], axis=0),
            np.stack([self.rew[i] for i in idx], axis=0),
            np.stack([self.nobs[i] for i in idx], axis=0),
            np.stack([self.done[i] for i in idx], axis=0),
        )


class SuccessReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int = 200_000, obs_dtype=np.float32, act_dtype=np.float32):
        super().__init__(capacity, obs_dtype, act_dtype)
        self.dists = deque(maxlen=self.capacity)

    def push(self, s, a, r, ns, d):
        super().push(s, a, r, ns, d)
        self.dists.append(np.float32(-1.0))

    def push_with_dist(self, s, a, r, ns, d, dist: Optional[float]):
        super().push(s, a, r, ns, d)
        self.dists.append(np.float32(-1.0 if dist is None else dist))

    def sample_by_dist(self, batch_size: int, min_dist: float = 0.0):
        if len(self) == 0:
            raise ValueError("SuccessReplayBuffer is empty.")
        if min_dist <= 0.0:
            return super().sample(batch_size)
        valid_idx = [i for i, dv in enumerate(self.dists) if (dv < 0.0) or (dv >= min_dist)]
        if not valid_idx:
            return super().sample(batch_size)
        replace = len(valid_idx) < batch_size
        choose = np.random.choice(valid_idx, size=batch_size, replace=replace)
        return (
            np.stack([self.obs[i] for i in choose], axis=0),
            np.stack([self.act[i] for i in choose], axis=0),
            np.stack([self.rew[i] for i in choose], axis=0),
            np.stack([self.nobs[i] for i in choose], axis=0),
            np.stack([self.done[i] for i in choose], axis=0),
        )


def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (768, 768, 768), log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean, log_std = torch.split(h, self.act_dim, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (log_std + 1.0) * (self.log_std_max - self.log_std_min)
        return mean, log_std

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        x_t = mean + std * noise
        a = torch.tanh(x_t)
        log_prob = (-0.5 * (((x_t - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-8).sum(dim=-1)
        return a, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (768, 768, 768)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.net(torch.cat([obs, act], dim=-1))


def init_bundle(obs_dim: int, act_dim: int, dev: torch.device, actor_lr: float, critic_lr: float, succ_buffer_capacity: int):
    actor = GaussianPolicy(obs_dim, act_dim).to(dev)
    critic_1 = QNetwork(obs_dim, act_dim).to(dev)
    critic_2 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_1 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_2 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())
    return {
        "actor": actor,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
        "actor_opt": optim.Adam(actor.parameters(), lr=actor_lr),
        "critic_1_opt": optim.Adam(critic_1.parameters(), lr=critic_lr),
        "critic_2_opt": optim.Adam(critic_2.parameters(), lr=critic_lr),
        "replay_buffer": ReplayBuffer(capacity=1_000_000),
        "succ_replay_buffer": SuccessReplayBuffer(capacity=succ_buffer_capacity),
        "log_alpha": nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32, device=dev)),
        "log_alpha_opt": None,  # filled below
        "alpha": 0.2,
        "target_entropy": -float(act_dim),
    }


def _finalize_bundle(bundle):
    bundle["log_alpha_opt"] = optim.Adam([bundle["log_alpha"]], lr=1e-5)
    return bundle


def policy_actions(bundle, obs_arr: np.ndarray, deterministic: bool = True) -> np.ndarray:
    actor = bundle["actor"]
    act_dim = actor.act_dim
    actions = np.zeros((obs_arr.shape[0], act_dim), dtype=np.float32)
    sensor_ok = obs_arr[:, -1] <= 0.5 if obs_arr.ndim >= 2 and obs_arr.shape[-1] > 0 else np.ones((obs_arr.shape[0],), dtype=bool)
    idxs = np.where(sensor_ok)[0]
    if idxs.size == 0:
        return actions
    s = to_tensor(np.asarray(obs_arr[idxs], dtype=np.float32), next(actor.parameters()).device)
    if deterministic:
        a = actor.act_deterministic(s).cpu().numpy()
    else:
        a, _ = actor.sample(s)
        a = a.detach().cpu().numpy()
    actions[idxs] = a
    return actions


def save_sac_checkpoint(path: str, bundle: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
    torch.save(
        {
            "format": "base_move_sac",
            "actor": bundle["actor"].state_dict(),
            "critic_1": bundle["critic_1"].state_dict(),
            "critic_2": bundle["critic_2"].state_dict(),
            "target_critic_1": bundle["target_critic_1"].state_dict(),
            "target_critic_2": bundle["target_critic_2"].state_dict(),
            "actor_opt": bundle["actor_opt"].state_dict(),
            "critic_1_opt": bundle["critic_1_opt"].state_dict(),
            "critic_2_opt": bundle["critic_2_opt"].state_dict(),
            "replay": {
                "obs": list(bundle["replay_buffer"].obs),
                "act": list(bundle["replay_buffer"].act),
                "rew": list(bundle["replay_buffer"].rew),
                "nobs": list(bundle["replay_buffer"].nobs),
                "done": list(bundle["replay_buffer"].done),
                "capacity": bundle["replay_buffer"].capacity,
            },
            "succ_replay": {
                "obs": list(bundle["succ_replay_buffer"].obs),
                "act": list(bundle["succ_replay_buffer"].act),
                "rew": list(bundle["succ_replay_buffer"].rew),
                "nobs": list(bundle["succ_replay_buffer"].nobs),
                "done": list(bundle["succ_replay_buffer"].done),
                "dists": list(bundle["succ_replay_buffer"].dists),
                "capacity": bundle["succ_replay_buffer"].capacity,
            },
            "alpha": float(bundle["alpha"]),
            "target_entropy": float(bundle["target_entropy"]),
            "extra": extra or {},
        },
        path,
    )


def save_actor_checkpoint(path: str, bundle: Dict[str, Any]):
    torch.save({"format": "base_move_actor", "actor": bundle["actor"].state_dict()}, path)


def load_sac_checkpoint(path: str, obs_dim: int, act_dim: int, device: Optional[torch.device] = None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if ckpt.get("format") != "base_move_sac":
        raise ValueError("Unsupported checkpoint format. Expected base_move_sac.")
    bundle = _finalize_bundle(init_bundle(obs_dim, act_dim, dev, actor_lr=3e-4, critic_lr=3e-4, succ_buffer_capacity=ckpt.get("succ_replay", {}).get("capacity", 200_000)))
    bundle["actor"].load_state_dict(ckpt["actor"])
    bundle["critic_1"].load_state_dict(ckpt["critic_1"])
    bundle["critic_2"].load_state_dict(ckpt["critic_2"])
    bundle["target_critic_1"].load_state_dict(ckpt["target_critic_1"])
    bundle["target_critic_2"].load_state_dict(ckpt["target_critic_2"])
    bundle["actor_opt"].load_state_dict(ckpt["actor_opt"])
    bundle["critic_1_opt"].load_state_dict(ckpt["critic_1_opt"])
    bundle["critic_2_opt"].load_state_dict(ckpt["critic_2_opt"])
    rb = ReplayBuffer(capacity=ckpt["replay"]["capacity"])
    for s, a, r, ns, d in zip(ckpt["replay"]["obs"], ckpt["replay"]["act"], ckpt["replay"]["rew"], ckpt["replay"]["nobs"], ckpt["replay"]["done"]):
        rb.push(s, a, r, ns, d)
    srb = SuccessReplayBuffer(capacity=ckpt["succ_replay"]["capacity"])
    for s, a, r, ns, d, dist in zip(ckpt["succ_replay"]["obs"], ckpt["succ_replay"]["act"], ckpt["succ_replay"]["rew"], ckpt["succ_replay"]["nobs"], ckpt["succ_replay"]["done"], ckpt["succ_replay"]["dists"]):
        srb.push_with_dist(s, a, r, ns, d, None if float(dist) < 0.0 else float(dist))
    bundle["replay_buffer"] = rb
    bundle["succ_replay_buffer"] = srb
    bundle["alpha"] = float(ckpt.get("alpha", 0.2))
    bundle["target_entropy"] = float(ckpt.get("target_entropy", -float(act_dim)))
    with torch.no_grad():
        bundle["log_alpha"].copy_(torch.tensor(np.log(bundle["alpha"]), dtype=torch.float32, device=dev))
    return {"bundle": bundle}


def sac_train(
    env,
    bundle: Optional[Dict[str, Any]] = None,
    succ_buffer_capacity: int = 200_000,
    episodes: int = 500,
    max_steps: int = 512,
    batch_size: int = 128,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    device: Optional[torch.device] = None,
    p_succ: float = 0.30,
    succ_gate_min: int = 2048,
    succ_ramp_cov: float = 0.25,
    updates_per_step: int = 2,
    alpha_floor: float = 0.05,
    alpha_ceiling: float = 1.00,
    alpha_freeze_recent: float | None = 0.40,
    alpha_freeze_succbuf: int = 150_000,
    alpha_fixed: float = 0.24,
    save_best_online: bool = True,
    best_delta: float = 0.02,
    best_min_episodes: int = 30,
    best_ckpt_path: str = "sac_best.pth",
    best_actor_path: str = "sac_actor_best.pth",
    last_ckpt_path: str = "sac_last.pth",
    last_actor_path: str = "sac_actor_last.pth",
    save_last_every_episodes: int = 10,
    succ_min_dist: float = 0.20,
    **kwargs,
):
    assert episodes > 0 and batch_size > 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = reset_env(env)
    obs_dim = infer_single_agent_obs_dim(env, probe)
    act_dim = infer_single_agent_act_dim(env)
    base_bundle = None if bundle is None else bundle.get("bundle")
    if base_bundle is None:
        base_bundle = _finalize_bundle(init_bundle(obs_dim, act_dim, device, actor_lr, critic_lr, succ_buffer_capacity))

    recent_terminal_counts: deque[Tuple[int, int]] = deque(maxlen=100)
    recent_detour_terminal_counts: deque[Tuple[int, int]] = deque(maxlen=100)
    recent_collision_counts: deque[int] = deque(maxlen=100)
    recent_detour_collision_counts: deque[int] = deque(maxlen=100)
    succ_buf_total_history: deque[int] = deque(maxlen=max(2, int(best_min_episodes)))
    best_score = -1.0
    best_collision_score = math.inf
    alpha_frozen = False
    best_snapshot = None
    base_move_success_horizon = max(1, int(kwargs.get("base_move_success_horizon", 8)))
    detour_reward_shaping = bool(kwargs.get("detour_reward_shaping", True))
    detour_lead_bonus = float(kwargs.get("detour_lead_bonus", 0.02))
    detour_collision_base_bonus = float(kwargs.get("detour_collision_base_bonus", 0.10))
    detour_collision_bonus_100 = float(kwargs.get("detour_collision_bonus_100", 0.10))
    detour_collision_bonus_50 = float(kwargs.get("detour_collision_bonus_50", 0.20))
    detour_collision_bonus_10 = float(kwargs.get("detour_collision_bonus_10", 0.50))
    detour_collision_bonus_0 = float(kwargs.get("detour_collision_bonus_0", 1.00))

    for ep in range(episodes):
        sampled_rules = maybe_sample_agent_role_rules(env)
        obs = reset_env(env)
        detour_ref_state = _init_detour_reference_state(env)
        detour_final_info: Dict[str, Any] = {"in_sense_mask": np.zeros((getattr(env, "num_agents", 0),), dtype=bool)}
        done = False
        ep_steps = 0
        ep_reward = 0.0
        ep_collision_total = 0
        ep_detour_collision_total = 0
        ep_initial_in_sense = 0
        ep_terminal_in_sense = 0
        ep_detour_terminal_in_sense = 0
        ep_total_agents = 0
        recent_success_traces: Dict[int, deque] = {}
        final_info: Dict[str, object] = {}
        if hasattr(env, "agent_positions") and hasattr(env, "goal_pos") and hasattr(env, "sense_radius"):
            init_dists = np.linalg.norm(np.asarray(env.goal_pos, dtype=np.float32)[None, :] - np.asarray(env.agent_positions, dtype=np.float32), axis=1)
            ep_total_agents = int(len(init_dists))
            ep_initial_in_sense = int(np.count_nonzero(init_dists <= float(env.sense_radius)))

        while (not done) and (ep_steps < (env.max_steps if max_steps is None else max_steps)):
            obs_arr = np.asarray(obs, dtype=np.float32)
            if _is_multi_agent_obs(obs_arr):
                act = policy_actions(base_bundle, obs_arr, deterministic=False)
            else:
                act = policy_actions(base_bundle, obs_arr.reshape(1, -1), deterministic=False).reshape(-1)

            next_obs, reward, done, info = step_env(env, act)
            ep_steps += 1
            reward_arr = np.asarray(reward, dtype=np.float32)
            next_obs_arr = np.asarray(next_obs, dtype=np.float32)
            if reward_arr.ndim == 0:
                reward_arr = reward_arr.reshape(1)
            if not _is_multi_agent_obs(obs_arr):
                obs_arr = obs_arr.reshape(1, -1)
                next_obs_arr = next_obs_arr.reshape(1, -1)
                act = np.asarray(act, dtype=np.float32).reshape(1, -1)

            actor_state_after_step = _snapshot_env_runtime_state(env)
            detour_final_info = {"in_sense_mask": np.zeros((obs_arr.shape[0],), dtype=bool)}
            if detour_reward_shaping:
                detour_final_info = _step_detour_reference(env, detour_ref_state)
                _apply_env_runtime_state(env, actor_state_after_step)
                actor_arrived = int(np.count_nonzero(np.asarray(getattr(env, "_arrived_agents", np.zeros((obs_arr.shape[0],), dtype=bool)), dtype=bool)))
                detour_arrived = int(np.count_nonzero(np.asarray(detour_ref_state.get("arrived_agents", np.zeros((obs_arr.shape[0],), dtype=bool)), dtype=bool)))
                if actor_arrived > detour_arrived and detour_lead_bonus > 0.0:
                    reward_arr = reward_arr + np.float32(detour_lead_bonus * float(actor_arrived - detour_arrived))
                detour_collided_arr = np.asarray(detour_final_info.get("collision_handled", np.zeros((obs_arr.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
                ep_detour_collision_total += int(np.count_nonzero(detour_collided_arr))

            final_info = info if isinstance(info, dict) else {}
            success_mask_arr = np.asarray(info.get("success_mask", np.zeros((obs_arr.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
            dist_values = np.asarray(info.get("dist_to_goal", np.full((obs_arr.shape[0],), -1.0, dtype=np.float32)), dtype=np.float32).reshape(-1)
            collided_arr = np.asarray(info.get("collided", np.zeros((obs_arr.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
            ep_collision_total += int(np.count_nonzero(collided_arr))
            ep_reward += float(np.mean(reward_arr))

            for agent_idx in range(obs_arr.shape[0]):
                dist = None if agent_idx >= len(dist_values) else float(dist_values[agent_idx])
                step_success = bool(agent_idx < len(success_mask_arr) and success_mask_arr[agent_idx])
                base_bundle["replay_buffer"].push(obs_arr[agent_idx], act[agent_idx], reward_arr[agent_idx], next_obs_arr[agent_idx], done)
                sample = (obs_arr[agent_idx], act[agent_idx], reward_arr[agent_idx], next_obs_arr[agent_idx], done, dist)
                if agent_idx not in recent_success_traces:
                    recent_success_traces[agent_idx] = deque(maxlen=base_move_success_horizon)
                recent_success_traces[agent_idx].append(sample)
                if step_success:
                    for s, a, r, ns, d, sd in recent_success_traces[agent_idx]:
                        if sd is None:
                            base_bundle["succ_replay_buffer"].push(s, a, r, ns, d)
                        else:
                            base_bundle["succ_replay_buffer"].push_with_dist(s, a, r, ns, d, sd)

            obs = next_obs

            replay_buffer = base_bundle["replay_buffer"]
            succ_replay_buffer = base_bundle["succ_replay_buffer"]
            if len(replay_buffer) < batch_size:
                continue

            actor = base_bundle["actor"]
            critic_1 = base_bundle["critic_1"]
            critic_2 = base_bundle["critic_2"]
            target_critic_1 = base_bundle["target_critic_1"]
            target_critic_2 = base_bundle["target_critic_2"]
            actor_opt = base_bundle["actor_opt"]
            critic_1_opt = base_bundle["critic_1_opt"]
            critic_2_opt = base_bundle["critic_2_opt"]
            log_alpha = base_bundle["log_alpha"]
            log_alpha_opt = base_bundle["log_alpha_opt"]
            target_entropy = base_bundle["target_entropy"]

            for _ in range(max(1, int(updates_per_step))):
                succ_size = len(succ_replay_buffer)
                succ_cov = succ_size / max(1, getattr(succ_replay_buffer, "capacity", succ_size))
                p_eff = 0.0 if succ_size < succ_gate_min else float(np.clip(p_succ, 0.0, 1.0)) * min(1.0, succ_cov / max(1e-6, succ_ramp_cov))
                k = max(0, min(int(round(batch_size * p_eff)), batch_size))
                if succ_size > 0 and k > 0:
                    S1, A1, R1, NS1, D1 = succ_replay_buffer.sample_by_dist(min(k, succ_size), min_dist=max(0.0, succ_min_dist))
                    k_eff = len(S1)
                else:
                    S1 = A1 = R1 = NS1 = D1 = None
                    k_eff = 0
                need = batch_size - k_eff
                S2, A2, R2, NS2, D2 = replay_buffer.sample(need)
                if k_eff > 0:
                    S = np.concatenate([S1, S2], axis=0)
                    A = np.concatenate([A1, A2], axis=0)
                    R = np.concatenate([R1, R2], axis=0)
                    NS = np.concatenate([NS1, NS2], axis=0)
                    D = np.concatenate([D1, D2], axis=0)
                else:
                    S, A, R, NS, D = S2, A2, R2, NS2, D2

                states = to_tensor(S, device)
                actions = to_tensor(A, device)
                rewards = to_tensor(R, device).unsqueeze(1)
                next_states = to_tensor(NS, device)
                dones = to_tensor(D, device).unsqueeze(1)

                with torch.no_grad():
                    next_a, next_logp = actor.sample(next_states)
                    tq1 = target_critic_1(next_states, next_a)
                    tq2 = target_critic_2(next_states, next_a)
                    target_q = torch.min(tq1, tq2) - log_alpha.exp() * next_logp.unsqueeze(1)
                    y = rewards + gamma * (1.0 - dones) * target_q

                q1 = critic_1(states, actions)
                q2 = critic_2(states, actions)
                critic_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
                critic_1_opt.zero_grad(set_to_none=True)
                critic_2_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_1_opt.step()
                critic_2_opt.step()

                pi, logp_pi = actor.sample(states)
                q_pi = torch.min(critic_1(states, pi), critic_2(states, pi))
                actor_loss = (log_alpha.exp() * logp_pi.unsqueeze(1) - q_pi).mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                if not alpha_frozen:
                    alpha_loss = (log_alpha * (-logp_pi.detach() - target_entropy)).mean()
                    log_alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    log_alpha_opt.step()
                    with torch.no_grad():
                        log_alpha.data.clamp_(min=float(np.log(max(1e-6, alpha_floor))), max=float(np.log(max(alpha_floor + 1e-6, alpha_ceiling))))
                else:
                    with torch.no_grad():
                        log_alpha.fill_(math.log(alpha_fixed))
                base_bundle["alpha"] = float(log_alpha.exp().item())
                soft_update_(critic_1, target_critic_1, tau)
                soft_update_(critic_2, target_critic_2, tau)

        terminal_mask = np.asarray(final_info.get("in_sense_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
        ep_total_agents = int(len(terminal_mask))
        ep_terminal_in_sense = int(np.count_nonzero(terminal_mask))
        detour_terminal_mask = np.asarray(detour_final_info.get("in_sense_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
        ep_detour_terminal_in_sense = int(np.count_nonzero(detour_terminal_mask))
        episode_collision_bonus = 0.0
        if detour_reward_shaping and ep_collision_total < ep_detour_collision_total:
            episode_collision_bonus = detour_collision_base_bonus
            if ep_collision_total == 0:
                episode_collision_bonus += detour_collision_bonus_0
            elif ep_collision_total <= 10:
                episode_collision_bonus += detour_collision_bonus_10
            elif ep_collision_total <= 50:
                episode_collision_bonus += detour_collision_bonus_50
            elif ep_collision_total <= 100:
                episode_collision_bonus += detour_collision_bonus_100
        if episode_collision_bonus > 0.0:
            last_agent_count = int(getattr(obs_arr, "shape", [0])[0]) if 'obs_arr' in locals() else 0
            for back_idx in range(1, last_agent_count + 1):
                try:
                    base_bundle["replay_buffer"].rew[-back_idx] = np.asarray(
                        np.asarray(base_bundle["replay_buffer"].rew[-back_idx], dtype=np.float32) + np.float32(episode_collision_bonus),
                        dtype=np.float32,
                    )
                except Exception:
                    break
            ep_reward += float(episode_collision_bonus)
        ep_terminal_rate = _relative_rate_vs_detour(ep_terminal_in_sense, ep_detour_terminal_in_sense)
        recent_terminal_counts.append((ep_terminal_in_sense, ep_total_agents))
        recent_detour_terminal_counts.append((ep_detour_terminal_in_sense, ep_total_agents))
        recent_collision_counts.append(int(ep_collision_total))
        recent_detour_collision_counts.append(int(ep_detour_collision_total))
        recent_terminal_in_sense = sum(s for s, _ in recent_terminal_counts)
        recent_terminal_agents = sum(a for _, a in recent_terminal_counts)
        recent_detour_terminal_in_sense = sum(s for s, _ in recent_detour_terminal_counts)
        recent_collision_total = sum(recent_collision_counts)
        recent_detour_collision_total = sum(recent_detour_collision_counts)
        recent_absolute_terminal_rate = 100.0 * recent_terminal_in_sense / max(1, recent_terminal_agents)
        recent_terminal_rate = _relative_rate_vs_detour(recent_terminal_in_sense, recent_detour_terminal_in_sense)
        succ_buf_total = len(base_bundle["succ_replay_buffer"])
        succ_buf_total_history.append(succ_buf_total)
        succ_buf_growth = succ_buf_total - succ_buf_total_history[0] if len(succ_buf_total_history) >= 2 else 0

        if not alpha_frozen and succ_buf_total >= alpha_freeze_succbuf:
            with torch.no_grad():
                base_bundle["log_alpha"].copy_(torch.tensor(math.log(alpha_fixed), device=base_bundle["log_alpha"].device))
                base_bundle["alpha"] = alpha_fixed
            alpha_frozen = True
            print(f"[α-FROZEN] succ_buf_total={succ_buf_total} | alpha_fixed={alpha_fixed:.3f}")

        if (ep + 1) % 10 == 0:
            print(
                f"[EP {ep + 1:5d}] steps={ep_steps:3d} R={ep_reward:8.2f} "
                f"| start_in_sense={ep_initial_in_sense:3d}/{ep_total_agents:3d} "
                f"| in_sense_end={ep_terminal_in_sense:3d}/{ep_total_agents:3d} "
                f"| detour_end={ep_detour_terminal_in_sense:3d}/{ep_total_agents:3d} ({ep_terminal_rate:5.1f}%) "
                f"| collisions={ep_collision_total:4d} detour_col={ep_detour_collision_total:4d} "
                f"| recent100={recent_terminal_in_sense:4d}/{recent_terminal_agents:4d} ({recent_absolute_terminal_rate:5.1f}%) "
                f"| detour100={recent_detour_terminal_in_sense:4d}/{recent_terminal_agents:4d} ({recent_terminal_rate:5.1f}%) "
                f"| recent100_col={recent_collision_total:4d} detour100_col={recent_detour_collision_total:4d} "
                f"| succ_buf={succ_buf_total} "
                f"| alpha={base_bundle['alpha']:.3f}"
                + (f" | agents={len(sampled_rules)}" if sampled_rules is not None else "")
            )

        if save_best_online and len(recent_terminal_counts) >= max(2, int(best_min_episodes)):
            min_rate_delta = float(best_delta)
            should_save_best = False
            if float(recent_absolute_terminal_rate) >= best_score + min_rate_delta:
                should_save_best = True
            elif abs(float(recent_absolute_terminal_rate) - best_score) <= min_rate_delta and int(recent_collision_total) < int(best_collision_score):
                should_save_best = True
            if should_save_best:
                best_score = float(recent_absolute_terminal_rate)
                best_collision_score = int(recent_collision_total)
                best_snapshot = {
                    "episodes": ep + 1,
                    "best_in_sense_end_rate": best_score,
                    "recent_in_sense_end_rate": recent_absolute_terminal_rate,
                    "recent_in_sense_end": recent_terminal_in_sense,
                    "recent_detour_in_sense_end": recent_detour_terminal_in_sense,
                    "recent_total_agents": recent_terminal_agents,
                    "recent_collision_total": int(recent_collision_total),
                    "succ_buf_total": succ_buf_total,
                }
                save_sac_checkpoint(best_ckpt_path, base_bundle, extra=best_snapshot)
                save_actor_checkpoint(best_actor_path, base_bundle)
                print(f"[BEST] ep={ep + 1} in_sense_end={recent_absolute_terminal_rate:.1f}% recent100_col={recent_collision_total} -> saved {best_actor_path}")

        if int(save_last_every_episodes) > 0 and ((ep + 1) % int(save_last_every_episodes) == 0):
            last_snapshot = {
                "episodes": ep + 1,
                "recent_in_sense_end_rate": recent_terminal_rate,
                "recent_in_sense_end": recent_terminal_in_sense,
                "recent_detour_in_sense_end": recent_detour_terminal_in_sense,
                "recent_total_agents": recent_terminal_agents,
                "succ_buf_total": succ_buf_total,
            }
            save_sac_checkpoint(last_ckpt_path, base_bundle, extra=last_snapshot)
            save_actor_checkpoint(last_actor_path, base_bundle)
            print(f"[LAST] ep={ep + 1} -> saved {last_actor_path}")

    last_snapshot = {
        "episodes": episodes,
        "recent_in_sense_end_rate": recent_terminal_rate if recent_terminal_counts else 0.0,
        "recent_in_sense_end": recent_terminal_in_sense if recent_terminal_counts else 0,
        "recent_detour_in_sense_end": recent_detour_terminal_in_sense if recent_detour_terminal_counts else 0,
        "recent_total_agents": recent_terminal_agents if recent_terminal_counts else 0,
        "succ_buf_total": len(base_bundle["succ_replay_buffer"]),
    }
    save_sac_checkpoint(last_ckpt_path, base_bundle, extra=last_snapshot)
    save_actor_checkpoint(last_actor_path, base_bundle)
    return {"bundle": base_bundle}
