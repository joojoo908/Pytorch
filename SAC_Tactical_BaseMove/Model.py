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
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (512, 512, 512), log_std_bounds=(-5.0, 2.0)):
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
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...] = (512, 512, 512)):
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
    succ_buf_total_history: deque[int] = deque(maxlen=max(2, int(best_min_episodes)))
    best_score = -1.0
    alpha_frozen = False
    best_snapshot = None
    base_move_success_horizon = max(1, int(kwargs.get("base_move_success_horizon", 8)))

    for ep in range(episodes):
        sampled_rules = maybe_sample_agent_role_rules(env)
        obs = reset_env(env)
        done = False
        ep_steps = 0
        ep_reward = 0.0
        ep_initial_in_sense = 0
        ep_terminal_in_sense = 0
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

            ep_reward += float(np.mean(reward_arr))
            final_info = info if isinstance(info, dict) else {}
            success_mask_arr = np.asarray(info.get("success_mask", np.zeros((obs_arr.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
            dist_values = np.asarray(info.get("dist_to_goal", np.full((obs_arr.shape[0],), -1.0, dtype=np.float32)), dtype=np.float32).reshape(-1)

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
        ep_terminal_rate = 100.0 * ep_terminal_in_sense / max(1, ep_total_agents)
        recent_terminal_counts.append((ep_terminal_in_sense, ep_total_agents))
        recent_terminal_in_sense = sum(s for s, _ in recent_terminal_counts)
        recent_terminal_agents = sum(a for _, a in recent_terminal_counts)
        recent_terminal_rate = 100.0 * recent_terminal_in_sense / max(1, recent_terminal_agents)
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
                f"| in_sense_end={ep_terminal_in_sense:3d}/{ep_total_agents:3d} ({ep_terminal_rate:5.1f}%) "
                f"| recent100={recent_terminal_in_sense:4d}/{recent_terminal_agents:4d} ({recent_terminal_rate:5.1f}%) "
                f"| succ_buf={succ_buf_total} growth@{len(succ_buf_total_history)}={succ_buf_growth} "
                f"| alpha={base_bundle['alpha']:.3f}"
                + (f" | agents={len(sampled_rules)}" if sampled_rules is not None else "")
            )

        if save_best_online and len(succ_buf_total_history) >= max(2, int(best_min_episodes)):
            min_growth_delta = max(1.0, float(best_delta))
            if float(succ_buf_growth) >= best_score + min_growth_delta:
                best_score = float(succ_buf_growth)
                best_snapshot = {
                    "episodes": ep + 1,
                    "best_succ_buf_growth": best_score,
                    "succ_buf_total": succ_buf_total,
                }
                save_sac_checkpoint(best_ckpt_path, base_bundle, extra=best_snapshot)
                save_actor_checkpoint(best_actor_path, base_bundle)
                print(f"[BEST] ep={ep + 1} growth={int(succ_buf_growth)} -> saved {best_actor_path}")

    save_actor_checkpoint("sac_actor_last.pth", base_bundle)
    return {"bundle": base_bundle}
