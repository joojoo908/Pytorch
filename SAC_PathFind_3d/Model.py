# Model.py
# SAC with success-replay mixing (gated & ramped), increased updates-per-step, alpha floor/ceiling,
# robust success detection fallbacks, optional minimum-distance filtering on success samples,
# gym/gymnasium API compatibility, and lightweight "best model" saving by recent online success.
#
# This file is self-contained and does not rely on a global `device` outside this module.

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Env API compatibility (Gym/Gymnasium)
# ----------------------------

def reset_env(env):
    """Return observation only. Supports gym (obs) and gymnasium (obs, info)."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out

def step_env(env, action):
    """Return (obs, reward, done, info). Supports gym (4-tuple) and gymnasium (5-tuple)."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, r, done, info
    elif isinstance(out, tuple) and len(out) == 4:
        return out
    raise RuntimeError("Unsupported env.step(...) return format")


def _is_multi_agent_obs(obs):
    arr = np.asarray(obs)
    return arr.ndim >= 2


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
    result = {
        "front": False,
        "cover": False,
        "base_move": False,
        "surround": False,
        "kiting": False,
    }
    if not isinstance(info, dict):
        return result
    role_ids = info.get("role_ids", None)
    success_mask = info.get("success_mask", None)
    if role_ids is None or success_mask is None:
        return result
    try:
        role_ids = np.asarray(role_ids, dtype=np.int32).reshape(-1)
        success_mask = np.asarray(success_mask, dtype=bool).reshape(-1)
    except Exception:
        return result
    for role_id, success in zip(role_ids, success_mask):
        if not bool(success):
            continue
        if int(role_id) == 0:
            result["front"] = True
        elif int(role_id) == 1:
            result["cover"] = True
        elif int(role_id) == 2:
            result["base_move"] = True
        elif int(role_id) == 3:
            result["surround"] = True
        elif int(role_id) == 4:
            result["kiting"] = True
    return result


def is_diverse_tactical_success(info: Dict[str, Any]) -> bool:
    role_success = extract_role_success(info)
    return bool(role_success["front"] and role_success["cover"] and role_success["surround"])


ROLE_ID_TO_NAME = {
    0: "front",
    1: "cover",
    2: "base_move",
    3: "surround",
    4: "kiting",
}
ROLE_IDS = tuple(sorted(ROLE_ID_TO_NAME.keys()))
ROLE_NONE = -1


def role_name(role_id: int) -> str:
    if int(role_id) == ROLE_NONE:
        return "none"
    return ROLE_ID_TO_NAME.get(int(role_id), f"role_{int(role_id)}")


def maybe_sample_agent_role_rules(env) -> Optional[List[str]]:
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


def get_env_role_ids(env, count: int) -> np.ndarray:
    role_ids = getattr(env, "agent_role_ids", None)
    if role_ids is None:
        return np.zeros((count,), dtype=np.int32)
    arr = np.asarray(role_ids, dtype=np.int32).reshape(-1)
    if arr.shape[0] < count:
        out = np.zeros((count,), dtype=np.int32)
        out[:arr.shape[0]] = arr
        return out
    return arr[:count]


# ----------------------------
# Small helpers
# ----------------------------

def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)

def soft_update_(src: nn.Module, dst: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def _push_success_sample(
    succ_replay_buffer,
    sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float]],
) -> None:
    s, a, r, ns, d, dist = sample
    if dist is None:
        succ_replay_buffer.push(s, a, r, ns, d)
    else:
        succ_replay_buffer.push_with_dist(s, a, r, ns, d, dist)


# ----------------------------
# Replay Buffers
# ----------------------------

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

    def clear(self):
        self.obs.clear(); self.act.clear(); self.rew.clear(); self.nobs.clear(); self.done.clear()

    def push(self, s, a, r, ns, d):
        self.obs.append(np.asarray(s, dtype=self._obs_dtype))
        self.act.append(np.asarray(a, dtype=self._act_dtype))
        self.rew.append(np.asarray(r, dtype=np.float32))
        self.nobs.append(np.asarray(ns, dtype=self._obs_dtype))
        self.done.append(np.asarray(d, dtype=np.float32))

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.obs), size=batch_size)
        S = np.stack([self.obs[i] for i in idx], axis=0)
        A = np.stack([self.act[i] for i in idx], axis=0)
        R = np.stack([self.rew[i] for i in idx], axis=0)
        NS = np.stack([self.nobs[i] for i in idx], axis=0)
        D = np.stack([self.done[i] for i in idx], axis=0)
        return S, A, R, NS, D


class SuccessReplayBuffer(ReplayBuffer):
    """
    Success-only buffer. Keeps an optional 'distance-to-goal' list to enable minimum-distance sampling.
    """
    def __init__(self, capacity: int = 200_000, obs_dtype=np.float32, act_dtype=np.float32):
        super().__init__(capacity, obs_dtype, act_dtype)
        self.dists = deque(maxlen=self.capacity)  # -1.0 means unknown/no-distance

    def push(self, s, a, r, ns, d):
        super().push(s, a, r, ns, d)
        self.dists.append(np.float32(-1.0))  # unknown

    def push_with_dist(self, s, a, r, ns, d, dist: Optional[float]):
        super().push(s, a, r, ns, d)
        if dist is None:
            self.dists.append(np.float32(-1.0))
        else:
            self.dists.append(np.float32(dist))

    def sample_by_dist(self, batch_size: int, min_dist: float = 0.0):
        if len(self) == 0:
            raise ValueError("SuccessReplayBuffer is empty.")
        if min_dist <= 0.0:
            # Same as ordinary sampling
            return super().sample(batch_size)

        valid_idx = []
        for i, dv in enumerate(self.dists):
            # Allow unknown distance (-1) or samples with distance >= min_dist
            if (dv < 0.0) or (dv >= min_dist):
                valid_idx.append(i)

        if len(valid_idx) == 0:
            # Fall back to ordinary sampling if all are too close
            return super().sample(batch_size)

        # Sample with replacement if needed
        replace = len(valid_idx) < batch_size
        choose = np.random.choice(valid_idx, size=batch_size if not replace else min(batch_size, len(valid_idx)), replace=replace)
        S = np.stack([self.obs[i] for i in choose], axis=0)
        A = np.stack([self.act[i] for i in choose], axis=0)
        R = np.stack([self.rew[i] for i in choose], axis=0)
        NS = np.stack([self.nobs[i] for i in choose], axis=0)
        D = np.stack([self.done[i] for i in choose], axis=0)
        return S, A, R, NS, D


# ----------------------------
# Networks
# ----------------------------

def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: Tuple[int, ...] = (512,512,512),
                 log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean, log_std = torch.split(h, self.act_dim, dim=-1)
        log_std = torch.tanh(log_std)  # [-1, 1]
        log_std = self.log_std_min + 0.5 * (log_std + 1.0) * (self.log_std_max - self.log_std_min)
        return mean, log_std

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        x_t = mean + std * noise
        a = torch.tanh(x_t)

        # log_prob with tanh correction
        log_prob = (-0.5 * (((x_t - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-8).sum(dim=-1)
        return a, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: Tuple[int, ...] = (512,512,512)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# ----------------------------
# Evaluation & Checkpoint
# ----------------------------
def _save_replay(rb: ReplayBuffer) -> Dict[str, Any]:
    return {"obs": list(rb.obs), "act": list(rb.act), "rew": list(rb.rew), "nobs": list(rb.nobs), "done": list(rb.done), "capacity": rb.capacity}


def _save_succ_replay(rb: SuccessReplayBuffer) -> Dict[str, Any]:
    return {
        "obs": list(rb.obs), "act": list(rb.act), "rew": list(rb.rew), "nobs": list(rb.nobs), "done": list(rb.done),
        "dists": list(rb.dists), "capacity": rb.capacity,
    }


def _load_replay(data: Dict[str, Any]) -> ReplayBuffer:
    rb = ReplayBuffer(capacity=data["capacity"])
    for s, a, r, ns, d in zip(data["obs"], data["act"], data["rew"], data["nobs"], data["done"]):
        rb.push(s, a, r, ns, d)
    return rb


def _load_succ_replay(data: Optional[Dict[str, Any]], capacity_default: int = 200_000) -> SuccessReplayBuffer:
    rb = SuccessReplayBuffer(capacity=(data["capacity"] if data is not None else capacity_default))
    if data is not None:
        for s, a, r, ns, d, dist in zip(data["obs"], data["act"], data["rew"], data["nobs"], data["done"], data["dists"]):
            rb.push_with_dist(s, a, r, ns, d, None if float(dist) < 0.0 else float(dist))
    return rb


def init_role_bundle(obs_dim: int, act_dim: int, dev: torch.device, actor_lr: float, critic_lr: float, succ_buffer_capacity: int) -> Dict[str, Any]:
    actor = GaussianPolicy(obs_dim, act_dim).to(dev)
    critic_1 = QNetwork(obs_dim, act_dim).to(dev)
    critic_2 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_1 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_2 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_1_opt = optim.Adam(critic_1.parameters(), lr=critic_lr)
    critic_2_opt = optim.Adam(critic_2.parameters(), lr=critic_lr)
    log_alpha = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32, device=dev))
    log_alpha_opt = optim.Adam([log_alpha], lr=1e-5)
    return {
        "actor": actor, "critic_1": critic_1, "critic_2": critic_2,
        "target_critic_1": target_critic_1, "target_critic_2": target_critic_2,
        "actor_opt": actor_opt, "critic_1_opt": critic_1_opt, "critic_2_opt": critic_2_opt,
        "replay_buffer": ReplayBuffer(capacity=1_000_000),
        "succ_replay_buffer": SuccessReplayBuffer(capacity=succ_buffer_capacity),
        "log_alpha": log_alpha, "log_alpha_opt": log_alpha_opt,
        "alpha": 0.2, "target_entropy": -float(act_dim),
    }


@torch.no_grad()
def role_policy_actions(role_bundles: Dict[int, Dict[str, Any]], obs_arr: np.ndarray, role_ids_arr: np.ndarray, deterministic: bool = True) -> np.ndarray:
    act_dim = next(iter(role_bundles.values()))["actor"].act_dim
    actions = np.zeros((obs_arr.shape[0], act_dim), dtype=np.float32)
    obs_arr = np.asarray(obs_arr, dtype=np.float32)
    sensor_ok = obs_arr[:, -1] <= 0.5 if obs_arr.ndim >= 2 and obs_arr.shape[-1] > 0 else np.ones((obs_arr.shape[0],), dtype=bool)
    for role_id in ROLE_IDS:
        idxs = np.where((role_ids_arr == role_id) & sensor_ok)[0]
        if idxs.size == 0:
            continue
        actor = role_bundles[int(role_id)]["actor"]
        device = next(actor.parameters()).device
        s = to_tensor(obs_arr[idxs], device)
        if deterministic:
            a = actor.act_deterministic(s).cpu().numpy()
        else:
            a, _ = actor.sample(s)
            a = a.detach().cpu().numpy()
        actions[idxs] = a
    return actions


@torch.no_grad()
def evaluate_success(env, role_bundles: Dict[int, Dict[str, Any]], episodes: int = 10, max_steps: Optional[int] = None, device: Optional[torch.device] = None) -> float:
    ok = 0
    for _ in range(episodes):
        obs = reset_env(env)
        done = False
        steps = 0
        info: Dict[str, Any] = {}
        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)
            if _is_multi_agent_obs(obs_arr):
                role_ids_arr = get_env_role_ids(env, obs_arr.shape[0])
                act = role_policy_actions(role_bundles, obs_arr, role_ids_arr, deterministic=True)
            else:
                role_ids_arr = get_env_role_ids(env, 1)
                act = role_policy_actions(role_bundles, obs_arr.reshape(1, -1), role_ids_arr, deterministic=True).reshape(-1)
            obs, _, done, info = step_env(env, act)
            steps += 1
            if (max_steps is not None) and (steps >= max_steps):
                break
        ok += int(is_diverse_tactical_success(info))
    return ok / float(max(1, episodes))


def save_sac_checkpoint(path: str, role_bundles: Dict[int, Dict[str, Any]], extra: Optional[Dict[str, Any]] = None, **kwargs):
    extra_dict = (extra or {}).copy()
    extra_dict.update(kwargs)
    roles_obj: Dict[str, Any] = {}
    actor_only: Dict[str, Any] = {}
    for role_id, bundle in role_bundles.items():
        key = role_name(role_id)
        roles_obj[key] = {
            "actor": bundle["actor"].state_dict(),
            "critic_1": bundle["critic_1"].state_dict(),
            "critic_2": bundle["critic_2"].state_dict(),
            "target_critic_1": bundle["target_critic_1"].state_dict(),
            "target_critic_2": bundle["target_critic_2"].state_dict(),
            "actor_opt": bundle["actor_opt"].state_dict(),
            "critic_1_opt": bundle["critic_1_opt"].state_dict(),
            "critic_2_opt": bundle["critic_2_opt"].state_dict(),
            "replay": _save_replay(bundle["replay_buffer"]),
            "succ_replay": _save_succ_replay(bundle["succ_replay_buffer"]),
            "alpha": float(bundle["alpha"]),
            "target_entropy": float(bundle["target_entropy"]),
        }
        actor_only[key] = bundle["actor"].state_dict()
    torch.save({"format": "multi_role_sac", "roles": roles_obj, "extra": extra_dict}, path)
    return actor_only


def _snapshot_role_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "actor": bundle["actor"].state_dict(),
        "critic_1": bundle["critic_1"].state_dict(),
        "critic_2": bundle["critic_2"].state_dict(),
        "target_critic_1": bundle["target_critic_1"].state_dict(),
        "target_critic_2": bundle["target_critic_2"].state_dict(),
        "actor_opt": bundle["actor_opt"].state_dict(),
        "critic_1_opt": bundle["critic_1_opt"].state_dict(),
        "critic_2_opt": bundle["critic_2_opt"].state_dict(),
        "replay": _save_replay(bundle["replay_buffer"]),
        "succ_replay": _save_succ_replay(bundle["succ_replay_buffer"]),
        "alpha": float(bundle["alpha"]),
        "target_entropy": float(bundle["target_entropy"]),
    }


def _save_best_role_snapshots(
    ckpt_path: str,
    actor_path: str,
    best_role_snapshots: Dict[int, Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    roles_obj: Dict[str, Any] = {}
    actor_only: Dict[str, Any] = {}
    for role_id in ROLE_IDS:
        key = role_name(role_id)
        snap = best_role_snapshots[role_id]
        roles_obj[key] = snap
        actor_only[key] = snap["actor"]
    torch.save({"format": "multi_role_sac", "roles": roles_obj, "extra": (extra or {})}, ckpt_path)
    torch.save({"format": "multi_role_actor", "actors": actor_only}, actor_path)


def load_sac_checkpoint(path: str, obs_dim: int, act_dim: int, device: Optional[torch.device] = None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if ckpt.get("format") != "multi_role_sac":
        raise ValueError("Unsupported checkpoint format. Expected multi_role_sac.")
    role_bundles: Dict[int, Dict[str, Any]] = {}
    for role_id in ROLE_IDS:
        key = role_name(role_id)
        role_data = ckpt["roles"][key]
        bundle = init_role_bundle(obs_dim, act_dim, dev, actor_lr=3e-4, critic_lr=3e-4, succ_buffer_capacity=(role_data.get("succ_replay") or {}).get("capacity", 200_000))
        bundle["actor"].load_state_dict(role_data["actor"])
        bundle["critic_1"].load_state_dict(role_data["critic_1"])
        bundle["critic_2"].load_state_dict(role_data["critic_2"])
        bundle["target_critic_1"].load_state_dict(role_data["target_critic_1"])
        bundle["target_critic_2"].load_state_dict(role_data["target_critic_2"])
        bundle["actor_opt"].load_state_dict(role_data["actor_opt"])
        bundle["critic_1_opt"].load_state_dict(role_data["critic_1_opt"])
        bundle["critic_2_opt"].load_state_dict(role_data["critic_2_opt"])
        bundle["replay_buffer"] = _load_replay(role_data["replay"])
        bundle["succ_replay_buffer"] = _load_succ_replay(role_data.get("succ_replay"))
        bundle["alpha"] = float(role_data.get("alpha", 0.2))
        bundle["target_entropy"] = float(role_data.get("target_entropy", -float(act_dim)))
        with torch.no_grad():
            bundle["log_alpha"].copy_(torch.tensor(np.log(bundle["alpha"]), dtype=torch.float32, device=dev))
        role_bundles[role_id] = bundle
    return {"role_bundles": role_bundles}


# ----------------------------
# SAC Training
# ----------------------------

def sac_train(env,
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
              succ_min_dist: float = 0.20,
              **kwargs):
    assert episodes > 0 and batch_size > 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _probe = reset_env(env)
    obs_dim = infer_single_agent_obs_dim(env, _probe)
    act_dim = infer_single_agent_act_dim(env)
    role_bundles = None if bundle is None else bundle.get("role_bundles")
    if role_bundles is None:
        role_bundles = {role_id: init_role_bundle(obs_dim, act_dim, device, actor_lr, critic_lr, succ_buffer_capacity) for role_id in ROLE_IDS}

    recent_role_step_counts = {role_id: deque(maxlen=100) for role_id in ROLE_IDS}
    recent_role_succbuf_totals = {role_id: deque(maxlen=max(2, int(best_min_episodes))) for role_id in ROLE_IDS}
    succ_buf_total_history: deque[int] = deque(maxlen=max(2, int(best_min_episodes)))
    best_score = -1.0
    best_role_scores = {role_id: -1.0 for role_id in ROLE_IDS}
    best_role_snapshots = {role_id: _snapshot_role_bundle(role_bundles[role_id]) for role_id in ROLE_IDS}
    alpha_frozen = False
    base_move_success_horizon = max(1, int(kwargs.get("base_move_success_horizon", 8)))

    for ep in range(episodes):
        sampled_rules = maybe_sample_agent_role_rules(env)
        obs = reset_env(env)
        done = False
        ep_steps = 0
        ep_reward = 0.0
        info: Dict[str, Any] = {}
        obs_arr0 = np.asarray(obs, dtype=np.float32)
        ep_horizon = (env.max_steps if (max_steps is None and hasattr(env, "max_steps")) else max_steps)
        ep_role_attempts = {role_id: 0 for role_id in ROLE_IDS}
        ep_role_successes = {role_id: 0 for role_id in ROLE_IDS}
        recent_success_traces: Dict[Tuple[int, int], deque] = {}

        while (not done) and (ep_steps < (ep_horizon if ep_horizon is not None else 10**9)):
            obs_arr = np.asarray(obs, dtype=np.float32)
            if _is_multi_agent_obs(obs_arr):
                role_ids_arr = get_env_role_ids(env, obs_arr.shape[0])
                act = role_policy_actions(role_bundles, obs_arr, role_ids_arr, deterministic=False)
            else:
                role_ids_arr = get_env_role_ids(env, 1)
                act = role_policy_actions(role_bundles, obs_arr.reshape(1, -1), role_ids_arr, deterministic=False).reshape(-1)

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

            ep_reward += float(np.mean(reward_arr))
            dist_values = None
            if isinstance(info, dict) and ("dist_to_goal" in info) and info["dist_to_goal"] is not None:
                try:
                    dist_values = np.asarray(info["dist_to_goal"], dtype=np.float32).reshape(-1)
                except Exception:
                    dist_values = None
            success_mask_arr = None
            if isinstance(info, dict) and ("success_mask" in info) and info["success_mask"] is not None:
                try:
                    success_mask_arr = np.asarray(info["success_mask"], dtype=bool).reshape(-1)
                except Exception:
                    success_mask_arr = None
            info_role_ids_arr = role_ids_arr
            if isinstance(info, dict) and ("role_ids" in info) and info["role_ids"] is not None:
                try:
                    info_role_ids_arr = np.asarray(info["role_ids"], dtype=np.int32).reshape(-1)
                except Exception:
                    info_role_ids_arr = role_ids_arr

            for agent_idx in range(obs_arr.shape[0]):
                role_id = int(info_role_ids_arr[agent_idx]) if agent_idx < len(info_role_ids_arr) else ROLE_NONE
                if role_id not in role_bundles:
                    continue
                dist = None if dist_values is None or agent_idx >= len(dist_values) else float(dist_values[agent_idx])
                step_success = bool(success_mask_arr is not None and agent_idx < len(success_mask_arr) and success_mask_arr[agent_idx])
                sample = (
                    obs_arr[agent_idx],
                    act[agent_idx],
                    reward_arr[agent_idx],
                    next_obs_arr[agent_idx],
                    done,
                    dist,
                )
                if role_id in ep_role_attempts:
                    ep_role_attempts[role_id] += 1
                    ep_role_successes[role_id] += int(step_success)
                role_bundles[role_id]["replay_buffer"].push(obs_arr[agent_idx], act[agent_idx], reward_arr[agent_idx], next_obs_arr[agent_idx], done)
                trace_key = (agent_idx, role_id)
                if trace_key not in recent_success_traces:
                    recent_success_traces[trace_key] = deque(maxlen=base_move_success_horizon)
                recent_success_traces[trace_key].append(sample)
                if step_success:
                    succ_replay_buffer = role_bundles[role_id]["succ_replay_buffer"]
                    if role_id == 2:
                        for trace_sample in recent_success_traces[trace_key]:
                            _push_success_sample(succ_replay_buffer, trace_sample)
                    else:
                        _push_success_sample(succ_replay_buffer, sample)

            obs = next_obs

            for role_id, rbundle in role_bundles.items():
                replay_buffer = rbundle["replay_buffer"]
                succ_replay_buffer = rbundle["succ_replay_buffer"]
                if len(replay_buffer) < batch_size:
                    continue
                actor = rbundle["actor"]
                critic_1 = rbundle["critic_1"]
                critic_2 = rbundle["critic_2"]
                target_critic_1 = rbundle["target_critic_1"]
                target_critic_2 = rbundle["target_critic_2"]
                actor_opt = rbundle["actor_opt"]
                critic_1_opt = rbundle["critic_1_opt"]
                critic_2_opt = rbundle["critic_2_opt"]
                log_alpha = rbundle["log_alpha"]
                log_alpha_opt = rbundle["log_alpha_opt"]
                target_entropy = rbundle["target_entropy"]

                for _ in range(max(1, int(updates_per_step))):
                    succ_size = len(succ_replay_buffer)
                    succ_cov = succ_size / max(1, getattr(succ_replay_buffer, "capacity", succ_size))
                    if succ_size < succ_gate_min:
                        p_eff = 0.0
                    else:
                        ramp = min(1.0, succ_cov / max(1e-6, succ_ramp_cov))
                        p_eff = float(np.clip(p_succ, 0.0, 1.0)) * ramp
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
                    q1_pi = critic_1(states, pi)
                    q2_pi = critic_2(states, pi)
                    q_pi = torch.min(q1_pi, q2_pi)
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
                    rbundle["alpha"] = float(log_alpha.exp().item())
                    soft_update_(critic_1, target_critic_1, tau)
                    soft_update_(critic_2, target_critic_2, tau)

        for role_id in ROLE_IDS:
            recent_role_step_counts[role_id].append((ep_role_successes[role_id], ep_role_attempts[role_id]))
        role_step_rates = {}
        role_step_counts = {}
        for role_id in ROLE_IDS:
            succ_count = sum(s for s, _ in recent_role_step_counts[role_id])
            attempt_count = sum(a for _, a in recent_role_step_counts[role_id])
            role_step_counts[role_id] = (succ_count, attempt_count)
            role_step_rates[role_id] = 100.0 * succ_count / max(1, attempt_count)
        succ_buf_total = sum(len(role_bundles[r]["succ_replay_buffer"]) for r in ROLE_IDS)
        succ_buf_total_history.append(succ_buf_total)
        role_succ_buf_totals = {}
        role_succ_buf_growths = {}
        for role_id in ROLE_IDS:
            role_total = len(role_bundles[role_id]["succ_replay_buffer"])
            role_succ_buf_totals[role_id] = role_total
            recent_role_succbuf_totals[role_id].append(role_total)
            role_succ_buf_growths[role_id] = (
                role_total - recent_role_succbuf_totals[role_id][0]
                if len(recent_role_succbuf_totals[role_id]) >= 2
                else 0
            )
        if len(succ_buf_total_history) >= 2:
            succ_buf_growth = succ_buf_total - succ_buf_total_history[0]
        else:
            succ_buf_growth = 0

        if not alpha_frozen:
            if succ_buf_total >= alpha_freeze_succbuf:
                for role_id in ROLE_IDS:
                    with torch.no_grad():
                        role_bundles[role_id]["log_alpha"].copy_(torch.tensor(math.log(alpha_fixed), device=role_bundles[role_id]["log_alpha"].device))
                        role_bundles[role_id]["alpha"] = alpha_fixed
                alpha_frozen = True
                print(f"[α-FROZEN] succ_buf_total={succ_buf_total} | alpha_fixed={alpha_fixed:.3f}")

        if (ep + 1) % 10 == 0:
            alpha_summary = "/".join(f"{role_name(r)}:{role_bundles[r]['alpha']:.3f}" for r in ROLE_IDS)
            succ_summary = "/".join(f"{role_name(r)}:{len(role_bundles[r]['succ_replay_buffer'])}" for r in ROLE_IDS)
            succ_growth_summary = "/".join(f"{role_name(r)}:{role_succ_buf_growths[r]}" for r in ROLE_IDS)
            role_step_summary = "/".join(f"{role_name(r)}:{role_step_rates[r]:4.1f}" for r in ROLE_IDS)
            role_step_count_summary = "/".join(f"{role_name(r)}:{role_step_counts[r][0]}/{role_step_counts[r][1]}" for r in ROLE_IDS)
            rule_summary = ""
            if sampled_rules is not None:
                rule_summary = f" | agents={len(sampled_rules)} rules={','.join(sampled_rules)}"
            print(f"[EP {ep+1:5d}] steps={ep_steps:3d}  R={ep_reward:8.2f}  "
                  f"| role_step={role_step_summary} "
                  f"| role_step_n={role_step_count_summary} "
                  f"| succ_buf_total={succ_buf_total} growth@{len(succ_buf_total_history)}={succ_buf_growth} "
                  f"| succ_growth={succ_growth_summary} "
                  f"| alpha={alpha_summary} | succ_buf={succ_summary}{rule_summary}")

        if save_best_online and len(succ_buf_total_history) >= max(2, int(best_min_episodes)):
            min_growth_delta = max(1.0, float(best_delta))
            if float(succ_buf_growth) >= best_score + min_growth_delta:
                best_score = float(succ_buf_growth)
            improved_roles = []
            for role_id in ROLE_IDS:
                role_growth = float(role_succ_buf_growths[role_id])
                if role_growth >= best_role_scores[role_id] + min_growth_delta:
                    best_role_scores[role_id] = role_growth
                    best_role_snapshots[role_id] = _snapshot_role_bundle(role_bundles[role_id])
                    improved_roles.append(f"{role_name(role_id)}:{int(role_growth)}")
            if improved_roles:
                _save_best_role_snapshots(
                    best_ckpt_path,
                    best_actor_path,
                    best_role_snapshots,
                    extra={
                        "best_succ_buf_growth_total": float(best_score),
                        "best_role_succ_buf_growth": {role_name(r): float(best_role_scores[r]) for r in ROLE_IDS},
                        "succ_buf_total": succ_buf_total,
                        "episodes": ep + 1,
                    },
                )
                print(f"[BEST-role] ep={ep+1} updated={','.join(improved_roles)} → saved {best_actor_path}")

    torch.save({"format": "multi_role_actor", "actors": {role_name(role_id): role_bundles[role_id]["actor"].state_dict() for role_id in ROLE_IDS}}, "sac_actor_last.pth")
    return {"role_bundles": role_bundles}
