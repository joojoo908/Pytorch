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
        "flank_left": False,
        "flank_right": False,
        "cover": False,
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
            result["flank_left"] = True
        elif int(role_id) == 2:
            result["flank_right"] = True
        elif int(role_id) == 3:
            result["cover"] = True
    return result


def is_diverse_tactical_success(info: Dict[str, Any]) -> bool:
    role_success = extract_role_success(info)
    flank_ok = role_success["flank_left"] or role_success["flank_right"]
    return bool(role_success["front"] and flank_ok and role_success["cover"])


# ----------------------------
# Small helpers
# ----------------------------

def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)

def soft_update_(src: nn.Module, dst: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


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

@torch.no_grad()
def evaluate_success(env, actor: GaussianPolicy, episodes: int = 10, max_steps: Optional[int] = None, device: Optional[torch.device] = None) -> float:
    """Deterministic policy evaluation (success rate in [0,1])."""
    if device is None:
        device = next(actor.parameters()).device
    ok = 0
    for _ in range(episodes):
        obs = reset_env(env)
        done = False
        steps = 0
        info: Dict[str, Any] = {}
        reward = 0.0
        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)
            if _is_multi_agent_obs(obs_arr):
                s = to_tensor(obs_arr, device)
                a = actor.act_deterministic(s).cpu().numpy()
            else:
                s = to_tensor(obs_arr, device).unsqueeze(0)
                a = actor.act_deterministic(s).squeeze(0).cpu().numpy()
            obs, reward, done, info = step_env(env, a)
            steps += 1
            if (max_steps is not None) and (steps >= max_steps):
                break

        # Success detection with fallbacks
        success = is_diverse_tactical_success(info)
        ok += int(success)
    return ok / float(max(1, episodes))


def save_sac_checkpoint(path: str,
                        actor: nn.Module, critic_1: nn.Module, critic_2: nn.Module,
                        target_critic_1: nn.Module, target_critic_2: nn.Module,
                        actor_opt: optim.Optimizer, critic_1_opt: optim.Optimizer, critic_2_opt: optim.Optimizer,
                        replay_buffer: ReplayBuffer,
                        alpha: float, target_entropy: float,
                        succ_replay_buffer: Optional[SuccessReplayBuffer] = None,
                        extra: Optional[Dict[str, Any]] = None,
                        **kwargs):
    """Save a minimal but sufficient checkpoint for resuming."""
    # absorb extra kwargs into extra field
    extra_dict = (extra or {}).copy()
    extra_dict.update(kwargs)

    obj = {
        "actor": actor.state_dict(),
        "critic_1": critic_1.state_dict(),
        "critic_2": critic_2.state_dict(),
        "target_critic_1": target_critic_1.state_dict(),
        "target_critic_2": target_critic_2.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "critic_1_opt": critic_1_opt.state_dict(),
        "critic_2_opt": critic_2_opt.state_dict(),
        "replay": {
            "obs": list(replay_buffer.obs),
            "act": list(replay_buffer.act),
            "rew": list(replay_buffer.rew),
            "nobs": list(replay_buffer.nobs),
            "done": list(replay_buffer.done),
            "capacity": replay_buffer.capacity,
        },
        # [PATCH] 성공 리플레이도 같이 저장해야 resume 시 성공 샘플 분포가 유지된다.
        "succ_replay": None if succ_replay_buffer is None else {
            "obs": list(succ_replay_buffer.obs),
            "act": list(succ_replay_buffer.act),
            "rew": list(succ_replay_buffer.rew),
            "nobs": list(succ_replay_buffer.nobs),
            "done": list(succ_replay_buffer.done),
            "dists": list(succ_replay_buffer.dists),
            "capacity": succ_replay_buffer.capacity,
        },
        "alpha": float(alpha),
        "target_entropy": float(target_entropy),
        "extra": extra_dict,
    }
    torch.save(obj, path)


def load_sac_checkpoint(path: str, obs_dim: int, act_dim: int, device: Optional[torch.device] = None):
    """Load a previously saved SAC checkpoint. Returns a bundle dict."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)

    actor = GaussianPolicy(obs_dim, act_dim).to(dev)
    critic_1 = QNetwork(obs_dim, act_dim).to(dev)
    critic_2 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_1 = QNetwork(obs_dim, act_dim).to(dev)
    target_critic_2 = QNetwork(obs_dim, act_dim).to(dev)

    actor.load_state_dict(ckpt["actor"])
    critic_1.load_state_dict(ckpt["critic_1"])
    critic_2.load_state_dict(ckpt["critic_2"])
    target_critic_1.load_state_dict(ckpt["target_critic_1"])
    target_critic_2.load_state_dict(ckpt["target_critic_2"])

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_1_opt = optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_opt = optim.Adam(critic_2.parameters(), lr=3e-4)
    actor_opt.load_state_dict(ckpt["actor_opt"])
    critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
    critic_2_opt.load_state_dict(ckpt["critic_2_opt"])

    rb = ReplayBuffer(capacity=ckpt["replay"]["capacity"])
    for s, a, r, ns, d in zip(ckpt["replay"]["obs"], ckpt["replay"]["act"], ckpt["replay"]["rew"], ckpt["replay"]["nobs"], ckpt["replay"]["done"]):
        rb.push(s, a, r, ns, d)

    # [PATCH] 성공 리플레이 버퍼도 복구한다. 없으면 빈 버퍼로 시작한다.
    succ_rb_data = ckpt.get("succ_replay", None)
    succ_rb = SuccessReplayBuffer(capacity=(succ_rb_data["capacity"] if succ_rb_data is not None else 200_000))
    if succ_rb_data is not None:
        for s, a, r, ns, d, dist in zip(
            succ_rb_data["obs"],
            succ_rb_data["act"],
            succ_rb_data["rew"],
            succ_rb_data["nobs"],
            succ_rb_data["done"],
            succ_rb_data["dists"],
        ):
            succ_rb.push_with_dist(s, a, r, ns, d, None if float(dist) < 0.0 else float(dist))

    alpha = float(ckpt.get("alpha", 0.2))
    target_entropy = float(ckpt.get("target_entropy", -float(act_dim)))

    return {
        "actor": actor, "critic_1": critic_1, "critic_2": critic_2,
        "target_critic_1": target_critic_1, "target_critic_2": target_critic_2,
        "actor_opt": actor_opt, "critic_1_opt": critic_1_opt, "critic_2_opt": critic_2_opt,
        "replay_buffer": rb,
        "succ_replay_buffer": succ_rb,
        "alpha": alpha, "target_entropy": target_entropy,
    }


# ----------------------------
# SAC Training
# ----------------------------

def sac_train(env,
              actor: Optional[GaussianPolicy] = None,
              critic_1: Optional[QNetwork] = None,
              critic_2: Optional[QNetwork] = None,
              target_critic_1: Optional[QNetwork] = None,
              target_critic_2: Optional[QNetwork] = None,
              actor_opt: Optional[optim.Optimizer] = None,
              critic_1_opt: Optional[optim.Optimizer] = None,
              critic_2_opt: Optional[optim.Optimizer] = None,
              replay_buffer: Optional[ReplayBuffer] = None,
              succ_replay_buffer: Optional[SuccessReplayBuffer] = None,
              # [PATCH] 런처에서 넘긴 checkpoint bundle / success buffer 용량을 실제로 사용한다.
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
              # ---- success mixing accel ----
              p_succ: float = 0.30,        # target mixing ratio upper bound
              succ_gate_min: int = 2048,   # below this size, don't mix at all
              succ_ramp_cov: float = 0.25, # ramp mixing by fill ratio up to this coverage
              # ---- UDR (updates per env step) ----
              updates_per_step: int = 2,
              # ---- exploration bounds ----
              alpha_floor: float = 0.05,
              alpha_ceiling: float = 1.00,
              # === NEW: alpha freeze options ===
              alpha_freeze_recent: float | None = 0.40,   # recent@100 성공률이 40%↑면 고정
              alpha_freeze_succbuf: int = 150_000,        # 성공버퍼가 이 만큼 차면 고정
              alpha_fixed: float = 0.24,                  # 고정할 α 값
              # ---- online best saving ----
              save_best_online: bool = True,
              best_delta: float = 0.02,
              best_min_episodes: int = 30,
              best_ckpt_path: str = "sac_best.pth",
              best_actor_path: str = "sac_actor_best.pth",
              # ---- success sample min distance ----
              succ_min_dist: float = 0.20,
              **kwargs):
    """
    Train SAC and return a bundle of components.
    """
    assert episodes > 0 and batch_size > 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer obs/act dimensions
    _probe = reset_env(env)
    obs_dim = infer_single_agent_obs_dim(env, _probe)
    act_dim = infer_single_agent_act_dim(env)

    if bundle is not None:
        # [PATCH] Test.py 에서 bundle 만 넘겨도 resume 가 실제로 되도록 복구한다.
        actor = actor if actor is not None else bundle.get("actor")
        critic_1 = critic_1 if critic_1 is not None else bundle.get("critic_1")
        critic_2 = critic_2 if critic_2 is not None else bundle.get("critic_2")
        target_critic_1 = target_critic_1 if target_critic_1 is not None else bundle.get("target_critic_1")
        target_critic_2 = target_critic_2 if target_critic_2 is not None else bundle.get("target_critic_2")
        actor_opt = actor_opt if actor_opt is not None else bundle.get("actor_opt")
        critic_1_opt = critic_1_opt if critic_1_opt is not None else bundle.get("critic_1_opt")
        critic_2_opt = critic_2_opt if critic_2_opt is not None else bundle.get("critic_2_opt")
        replay_buffer = replay_buffer if replay_buffer is not None else bundle.get("replay_buffer")
        succ_replay_buffer = succ_replay_buffer if succ_replay_buffer is not None else bundle.get("succ_replay_buffer")

    # networks
    if actor is None:
        actor = GaussianPolicy(obs_dim, act_dim).to(device)
    if critic_1 is None:
        critic_1 = QNetwork(obs_dim, act_dim).to(device)
    if critic_2 is None:
        critic_2 = QNetwork(obs_dim, act_dim).to(device)
    if target_critic_1 is None:
        target_critic_1 = QNetwork(obs_dim, act_dim).to(device)
        target_critic_1.load_state_dict(critic_1.state_dict())
    if target_critic_2 is None:
        target_critic_2 = QNetwork(obs_dim, act_dim).to(device)
        target_critic_2.load_state_dict(critic_2.state_dict())

    if actor_opt is None:
        actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    if critic_1_opt is None:
        critic_1_opt = optim.Adam(critic_1.parameters(), lr=critic_lr)
    if critic_2_opt is None:
        critic_2_opt = optim.Adam(critic_2.parameters(), lr=critic_lr)

    if replay_buffer is None:
        replay_buffer = ReplayBuffer(capacity=1_000_000)
    if succ_replay_buffer is None:
        # [PATCH] CLI 의 --succ-buffer-cap 이 실제 버퍼 크기에 반영되도록 수정.
        succ_replay_buffer = SuccessReplayBuffer(capacity=succ_buffer_capacity)

    # alpha auto-tuning
    init_alpha = 0.2 if bundle is None else float(bundle.get("alpha", 0.2))
    target_entropy = -float(act_dim) if bundle is None else float(bundle.get("target_entropy", -float(act_dim)))
    # [PATCH] resume 시 alpha 도 같이 이어받는다.
    log_alpha = nn.Parameter(torch.tensor(np.log(init_alpha), dtype=torch.float32, device=device))
    log_alpha_opt = optim.Adam([log_alpha], lr=1e-5)
    alpha = float(log_alpha.exp().item())

    recent_success: deque[int] = deque(maxlen=100)
    recent_front_success: deque[int] = deque(maxlen=100)
    recent_flank_left_success: deque[int] = deque(maxlen=100)
    recent_flank_right_success: deque[int] = deque(maxlen=100)
    recent_cover_success: deque[int] = deque(maxlen=100)
    best_score = -1.0

    alpha_frozen = False

    for ep in range(episodes):
        obs = reset_env(env)
        done = False
        ep_steps = 0
        ep_reward = 0.0
        episode_traj: List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, Optional[float]]]] = []
        info: Dict[str, Any] = {}
        last_reward = 0.0
        obs_arr0 = np.asarray(obs, dtype=np.float32)
        n_agents = int(obs_arr0.shape[0]) if _is_multi_agent_obs(obs_arr0) else 1
        episode_traj = [[] for _ in range(n_agents)]

        ep_horizon = (env.max_steps if (max_steps is None and hasattr(env, "max_steps"))
                      else max_steps)
        while (not done) and (ep_steps < (ep_horizon if ep_horizon is not None else 10**9)):
            obs_arr = np.asarray(obs, dtype=np.float32)
            if _is_multi_agent_obs(obs_arr):
                s = to_tensor(obs_arr, device)
                act_t, logp_t = actor.sample(s)
                act = act_t.detach().cpu().numpy()
            else:
                s = to_tensor(obs_arr, device).unsqueeze(0)
                act_t, logp_t = actor.sample(s)
                act = act_t.squeeze(0).detach().cpu().numpy()

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
            last_reward = float(np.mean(reward_arr))

            dist_values = None
            if isinstance(info, dict) and ("dist_to_goal" in info) and info["dist_to_goal"] is not None:
                try:
                    dist_values = np.asarray(info["dist_to_goal"], dtype=np.float32).reshape(-1)
                except Exception:
                    dist_values = None

            for agent_idx in range(obs_arr.shape[0]):
                dist = None if dist_values is None or agent_idx >= len(dist_values) else float(dist_values[agent_idx])
                replay_buffer.push(obs_arr[agent_idx], act[agent_idx], reward_arr[agent_idx], next_obs_arr[agent_idx], done)
                episode_traj[agent_idx].append((
                    np.asarray(obs_arr[agent_idx], dtype=np.float32),
                    np.asarray(act[agent_idx], dtype=np.float32),
                    float(reward_arr[agent_idx]),
                    np.asarray(next_obs_arr[agent_idx], dtype=np.float32),
                    bool(done),
                    dist
                ))

            obs = next_obs

            # ---- updates per step ----
            if len(replay_buffer) >= batch_size:
                for _ in range(max(1, int(updates_per_step))):
                    # compute effective success mixing ratio (gate & ramp)
                    succ_size = len(succ_replay_buffer)
                    succ_cov = succ_size / max(1, getattr(succ_replay_buffer, "capacity", succ_size))
                    if succ_size < succ_gate_min:
                        p_eff = 0.0
                    else:
                        ramp = min(1.0, succ_cov / max(1e-6, succ_ramp_cov))
                        p_eff = float(np.clip(p_succ, 0.0, 1.0)) * ramp
                        if len(recent_success) >= 10:
                            rec_frac = (sum(recent_success) / len(recent_success))
                            if rec_frac < 0.5:
                                p_eff *= 0.7  # early reduce

                    k = int(round(batch_size * p_eff))
                    k = max(0, min(k, batch_size))

                    # sample success part with minimum-distance filter
                    if succ_size > 0 and k > 0:
                        S1, A1, R1, NS1, D1 = succ_replay_buffer.sample_by_dist(
                            min(k, succ_size), min_dist=max(0.0, succ_min_dist)
                        )
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

                    states      = to_tensor(S, device)
                    actions     = to_tensor(A, device)
                    rewards     = to_tensor(R, device).unsqueeze(1)
                    next_states = to_tensor(NS, device)
                    dones       = to_tensor(D, device).unsqueeze(1)

                    # ----- critic update -----
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

                    # ----- actor update -----
                    pi, logp_pi = actor.sample(states)
                    q1_pi = critic_1(states, pi)
                    q2_pi = critic_2(states, pi)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (log_alpha.exp() * logp_pi.unsqueeze(1) - q_pi).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()

                    # ----- alpha update (+ optional freeze) -----
                    if not alpha_frozen:
                        alpha_loss = (log_alpha * (-logp_pi.detach() - target_entropy)).mean()
                        log_alpha_opt.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        log_alpha_opt.step()
                        with torch.no_grad():
                            lo = float(np.log(max(1e-6, alpha_floor)))
                            hi = float(np.log(max(alpha_floor + 1e-6, alpha_ceiling)))
                            log_alpha.data.clamp_(min=lo, max=hi)
                            alpha = float(log_alpha.exp().item())
                    else:
                        # keep α strictly fixed
                        with torch.no_grad():
                            log_alpha.fill_(math.log(alpha_fixed))
                            alpha = float(log_alpha.exp().item())

                    # ----- target networks -----
                    soft_update_(critic_1, target_critic_1, tau)
                    soft_update_(critic_2, target_critic_2, tau)

        # ---- episode end: success detection (fallbacks) ----
        role_success = extract_role_success(info)
        episode_success = is_diverse_tactical_success(info)

        if episode_success:
            for agent_traj in episode_traj:
                for (s_, a_, r_, ns_, d_, dist_) in agent_traj:
                    if dist_ is None:
                        succ_replay_buffer.push(s_, a_, r_, ns_, d_)
                    else:
                        succ_replay_buffer.push_with_dist(s_, a_, r_, ns_, d_, dist_)

        recent_success.append(1 if episode_success else 0)
        recent_front_success.append(1 if role_success["front"] else 0)
        recent_flank_left_success.append(1 if role_success["flank_left"] else 0)
        recent_flank_right_success.append(1 if role_success["flank_right"] else 0)
        recent_cover_success.append(1 if role_success["cover"] else 0)
        recent_rate = 100.0 * (sum(recent_success) / max(1, len(recent_success)))
        front_rate = 100.0 * (sum(recent_front_success) / max(1, len(recent_front_success)))
        flank_left_rate = 100.0 * (sum(recent_flank_left_success) / max(1, len(recent_flank_left_success)))
        flank_right_rate = 100.0 * (sum(recent_flank_right_success) / max(1, len(recent_flank_right_success)))
        cover_rate = 100.0 * (sum(recent_cover_success) / max(1, len(recent_cover_success)))

        # === NEW: freeze trigger ===
        if (not alpha_frozen) and (alpha_freeze_recent is not None):
            cond_recent = (len(recent_success) >= 50) and (recent_rate >= 100.0 * alpha_freeze_recent)
            cond_succbuf = (len(succ_replay_buffer) >= alpha_freeze_succbuf)
            if cond_recent and cond_succbuf:
                with torch.no_grad():
                    log_alpha.copy_(torch.tensor(math.log(alpha_fixed), device=log_alpha.device))
                alpha_frozen = True
                print(f"[α-FROZEN] recent@{len(recent_success)}={recent_rate:.1f}% "
                      f"| succ_buf={len(succ_replay_buffer)} | alpha_fixed={alpha_fixed:.3f}")

        if (ep + 1) % 10 == 0:
            print(f"[EP {ep+1:5d}] steps={ep_steps:3d}  R={ep_reward:8.2f}  succ={int(episode_success)}  "
                  f"recent@{len(recent_success)}={recent_rate:5.1f}%  "
                  f"| role(front/fl/flr/cov)={front_rate:4.1f}/{flank_left_rate:4.1f}/{flank_right_rate:4.1f}/{cover_rate:4.1f} "
                  f"| alpha={alpha:.3f} | succ_buf={len(succ_replay_buffer)}")

        # zero-cost best-by-online metric
        if save_best_online and len(recent_success) >= best_min_episodes:
            recent_mean = sum(recent_success) / len(recent_success)
            if recent_mean >= best_score + best_delta:
                best_score = float(recent_mean)
                save_sac_checkpoint(best_ckpt_path, actor, critic_1, critic_2,
                                    target_critic_1, target_critic_2,
                                    actor_opt, critic_1_opt, critic_2_opt,
                                    replay_buffer, alpha, target_entropy,
                                    succ_replay_buffer=succ_replay_buffer,
                                    extra={"best_score": best_score, "episodes": ep + 1})
                torch.save(actor.state_dict(), best_actor_path)
                print(f"[BEST-online] ep={ep+1} mean={recent_mean:.3f} → saved {best_actor_path}")

    # Save final actor (optional)
    torch.save(actor.state_dict(), "sac_actor_last.pth")
    return {
        "actor": actor,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
        "actor_opt": actor_opt,
        "critic_1_opt": critic_1_opt,
        "critic_2_opt": critic_2_opt,
        "replay_buffer": replay_buffer,
        "succ_replay_buffer": succ_replay_buffer,
        "alpha": alpha,
        "target_entropy": target_entropy,
    }
