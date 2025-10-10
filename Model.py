# Model.py — SAC (+BC 사전학습 연동) with 성공률 기반 target_entropy 보정 & Early-Stop
import os
import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================
# 기본 네트워크
# ======================================
def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
        elif out_act is not None:
            layers += [out_act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """
    tanh-가우시안 정책.
    sample(states) -> (action in [-1,1], log_prob (B,1))
    forward(states) -> (mean, log_std)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([state_dim] + list(hidden), nn.ReLU)
        self.mu = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Linear(hidden[-1], action_dim)
        self.LOG_STD_MIN = -20.0
        self.LOG_STD_MAX = 2.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> np.ndarray:
        a, _ = self.sample(x.unsqueeze(0))
        return a.cpu().numpy()[0]

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        # reparameterization
        eps = torch.randn_like(std)
        pre_tanh = mu + eps * std
        a = torch.tanh(pre_tanh)

        # log_prob for tanh-squashed Gaussian (stable)
        # log_prob(u) - sum(log(1 - tanh(u)^2))
        log_prob_u = (-0.5 * ((pre_tanh - mu) / (std + 1e-8)) ** 2
                      - log_std
                      - 0.5 * math.log(2 * math.pi))
        log_prob_u = log_prob_u.sum(dim=-1, keepdim=True)

        # stable: log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        correction = correction.sum(dim=-1, keepdim=True)
        log_prob = log_prob_u - correction
        return a, log_prob


class QNetwork(nn.Module):
    """Q(s,a)"""
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([state_dim + action_dim] + list(hidden) + [1], nn.ReLU)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


# ======================================
# 리플레이 버퍼
# ======================================
class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.s = []
        self.a = []
        self.r = []
        self.ns = []
        self.d = []

    def push(self, s, a, r, ns, done):
        if self.size < self.capacity:
            self.s.append(s)
            self.a.append(a)
            self.r.append(r)
            self.ns.append(ns)
            self.d.append(done)
            self.size += 1
        else:
            i = self.ptr
            self.s[i] = s
            self.a[i] = a
            self.r[i] = r
            self.ns[i] = ns
            self.d[i] = done
        self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self):
        return self.size

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        s = np.array([self.s[i] for i in idx], dtype=np.float32)
        a = np.array([self.a[i] for i in idx], dtype=np.float32)
        r = np.array([self.r[i] for i in idx], dtype=np.float32)
        ns = np.array([self.ns[i] for i in idx], dtype=np.float32)
        d = np.array([self.d[i] for i in idx], dtype=np.float32)
        return s, a, r, ns, d

    # ---- 직렬화/역직렬화 ----
    def to_numpy_dict(self, limit: int = 200_000):
        n = min(self.size, limit)
        start = max(0, self.size - n)
        return dict(
            s=np.array(self.s[start:self.size], dtype=np.float32),
            a=np.array(self.a[start:self.size], dtype=np.float32),
            r=np.array(self.r[start:self.size], dtype=np.float32),
            ns=np.array(self.ns[start:self.size], dtype=np.float32),
            d=np.array(self.d[start:self.size], dtype=np.float32),
        )

    @staticmethod
    def load_from_numpy_dict(data: dict, capacity: int = 1_000_000):
        buf = ReplayBuffer(capacity)
        if data is None:
            return buf
        S = data.get("s", None)
        if S is None:
            return buf
        n = len(S)
        for i in range(n):
            buf.push(data["s"][i], data["a"][i], float(data["r"][i]), data["ns"][i], bool(data["d"][i]))
        return buf


# ======================================
# 체크포인트 저장/로드
# ======================================
def save_sac_checkpoint(path: str,
                        actor: nn.Module,
                        critic_1: nn.Module, critic_2: nn.Module,
                        target_critic_1: nn.Module, target_critic_2: nn.Module,
                        actor_opt: optim.Optimizer,
                        critic_1_opt: optim.Optimizer, critic_2_opt: optim.Optimizer,
                        replay_buffer: Optional[ReplayBuffer] = None,
                        log_alpha: Optional[torch.Tensor] = None,
                        alpha_opt_state: Optional[dict] = None,
                        target_entropy: Optional[float] = None,
                        alpha_mode: Optional[str] = None,
                        fixed_alpha: Optional[float] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "critic_1": critic_1.state_dict(),
        "critic_2": critic_2.state_dict(),
        "target_critic_1": target_critic_1.state_dict(),
        "target_critic_2": target_critic_2.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "critic_1_opt": critic_1_opt.state_dict(),
        "critic_2_opt": critic_2_opt.state_dict(),
        "replay": (replay_buffer.to_numpy_dict() if replay_buffer is not None else None),
        "log_alpha": (float(log_alpha.detach().cpu().item()) if log_alpha is not None else None),
        "alpha_opt": alpha_opt_state,
        "target_entropy": (float(target_entropy) if target_entropy is not None else None),
        "alpha_mode": alpha_mode,
        "fixed_alpha": (None if fixed_alpha is None else float(fixed_alpha)),
    }, path)


def load_sac_checkpoint(path: str, state_dim: int, action_dim: int):
    ckpt = torch.load(path, map_location=device)
    actor = GaussianPolicy(state_dim, action_dim).to(device)
    critic_1 = QNetwork(state_dim, action_dim).to(device)
    critic_2 = QNetwork(state_dim, action_dim).to(device)
    target_critic_1 = QNetwork(state_dim, action_dim).to(device)
    target_critic_2 = QNetwork(state_dim, action_dim).to(device)

    actor.load_state_dict(ckpt["actor"])
    critic_1.load_state_dict(ckpt["critic_1"])
    critic_2.load_state_dict(ckpt["critic_2"])
    target_critic_1.load_state_dict(ckpt["target_critic_1"])
    target_critic_2.load_state_dict(ckpt["target_critic_2"])

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_1_opt = optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_opt = optim.Adam(critic_2.parameters(), lr=3e-4)
    if ckpt.get("actor_opt") is not None:
        actor_opt.load_state_dict(ckpt["actor_opt"])
    if ckpt.get("critic_1_opt") is not None:
        critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
    if ckpt.get("critic_2_opt") is not None:
        critic_2_opt.load_state_dict(ckpt["critic_2_opt"])

    replay = ReplayBuffer.load_from_numpy_dict(ckpt.get("replay"))

    # α 상태 복구
    la = ckpt.get("log_alpha")
    if la is None:
        log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=True)
    else:
        log_alpha = torch.tensor(float(la), device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    if ckpt.get("alpha_opt") is not None:
        alpha_opt.load_state_dict(ckpt["alpha_opt"])

    target_entropy = ckpt.get("target_entropy", None)
    alpha_mode = ckpt.get("alpha_mode", "auto")
    fixed_alpha = ckpt.get("fixed_alpha", None)

    return {
        "actor": actor,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
        "actor_opt": actor_opt,
        "critic_1_opt": critic_1_opt,
        "critic_2_opt": critic_2_opt,
        "replay_buffer": replay,
        "log_alpha": log_alpha,
        "alpha_opt": alpha_opt,
        "target_entropy": target_entropy,
        "alpha_mode": alpha_mode,
        "fixed_alpha": fixed_alpha,
    }


# ======================================
# SAC 학습 (성공률 기반 target_entropy 보정 + Early Stop)
# ======================================
def sac_train(env,
              actor=None,
              critic_1=None, critic_2=None,
              target_critic_1=None, target_critic_2=None,
              actor_opt=None, critic_1_opt=None, critic_2_opt=None,
              replay_buffer: Optional[ReplayBuffer] = None,
              episodes=500, batch_size=64, gamma=0.99, tau=0.005,
              # α 자동튜닝 상태
              log_alpha: Optional[torch.Tensor] = None,
              alpha_opt: Optional[optim.Optimizer] = None,
              target_entropy: Optional[float] = None,
              # α 제어
              auto_alpha=True, fixed_alpha=None,
              alpha_min=0.03, alpha_max=0.30,
              freeze_alpha_success: Optional[float] = None,
              # Early-Stop 옵션
              early_stop_success: Optional[float] = None,
              early_stop_patience: int = 3,
              early_stop_min_episodes: int = 300,
              # B안: 성공률 → target_entropy 보정
              success_target=0.90,
              te_lr=0.30,
              te_min: Optional[float] = None,
              te_max: Optional[float] = None):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = (actor or GaussianPolicy(state_dim, action_dim)).to(device)
    critic_1 = (critic_1 or QNetwork(state_dim, action_dim)).to(device)
    critic_2 = (critic_2 or QNetwork(state_dim, action_dim)).to(device)

    if target_critic_1 is None:
        target_critic_1 = QNetwork(state_dim, action_dim).to(device)
        target_critic_1.load_state_dict(critic_1.state_dict())
    else:
        target_critic_1 = target_critic_1.to(device)
    if target_critic_2 is None:
        target_critic_2 = QNetwork(state_dim, action_dim).to(device)
        target_critic_2.load_state_dict(critic_2.state_dict())
    else:
        target_critic_2 = target_critic_2.to(device)

    actor_opt = actor_opt or optim.Adam(actor.parameters(), lr=3e-4)
    critic_1_opt = critic_1_opt or optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_opt = critic_2_opt or optim.Adam(critic_2.parameters(), lr=3e-4)
    buffer = replay_buffer if (replay_buffer is not None) else ReplayBuffer()

    # α 자동튜닝 기본값
    if target_entropy is None:
        target_entropy = -float(action_dim)
    if log_alpha is None:
        log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=True)
    if alpha_opt is None:
        alpha_opt = optim.Adam([log_alpha], lr=3e-4)

    # 성공률/알파 상태
    recent_success = deque(maxlen=100)
    alpha_frozen = False
    _early_stop_streak = 0

    # B안: te 범위
    te_min = (-2.0 * float(action_dim)) if te_min is None else float(te_min)
    te_max = (-0.05 * float(action_dim)) if te_max is None else float(te_max)
    success_target = float(success_target)
    te_lr = float(te_lr)

    def current_alpha_value():
        if auto_alpha and not alpha_frozen:
            a = float(torch.clamp(log_alpha.exp(), min=alpha_min, max=alpha_max).item())
        else:
            a = fixed_alpha if fixed_alpha is not None else float(np.exp(float(log_alpha.detach().cpu().item())))
            a = max(alpha_min, min(alpha_max, a))
        return a

    success_count = 0

    for ep in range(int(episodes)):
        state, _ = env.reset()
        state = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
        total_reward = 0.0
        episode_success = False
        last_truncated = False
        last_reason = None

        max_steps = getattr(env, "max_steps", 300)
        for _ in range(max_steps):
            with torch.no_grad():
                action, _ = actor.sample(state.unsqueeze(0))
            action_np = action.cpu().numpy()[0]

            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            # 종료 사유 기록
            if isinstance(info, dict):
                terms = info.get("reward_terms") or {}
                if terms.get("success", 0):
                    episode_success = True
                    last_reason = "success"
                elif terms.get("collision_reset", False):
                    last_reason = "collision"

            buffer.push(state.cpu().numpy(), action_np, float(reward), np.array(next_state), done)

            state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
            total_reward += float(reward)

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                a_val = current_alpha_value()
                alpha_tensor = torch.as_tensor(a_val, dtype=torch.float32, device=device)

                # ---- Critic update ----
                with torch.no_grad():
                    next_action, log_prob_next = actor.sample(next_states)
                    tq1 = target_critic_1(next_states, next_action)
                    tq2 = target_critic_2(next_states, next_action)
                    target_q = torch.min(tq1, tq2) - alpha_tensor * log_prob_next
                    target_v = rewards + gamma * (1.0 - dones) * target_q

                q1 = critic_1(states, actions)
                q2 = critic_2(states, actions)
                critic_1_loss = F.mse_loss(q1, target_v)
                critic_2_loss = F.mse_loss(q2, target_v)

                critic_1_opt.zero_grad(); critic_1_loss.backward(); critic_1_opt.step()
                critic_2_opt.zero_grad(); critic_2_loss.backward(); critic_2_opt.step()

                # ---- Actor update ----
                new_action, log_prob = actor.sample(states)
                q1_new = critic_1(states, new_action)
                q2_new = critic_2(states, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha_tensor * log_prob - q_new).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                # ---- α update ----
                if auto_alpha and not alpha_frozen:
                    alpha_loss = (log_alpha.exp() * (-log_prob - target_entropy).detach()).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                # ---- Polyak ----
                with torch.no_grad():
                    for tp, p in zip(target_critic_1.parameters(), critic_1.parameters()):
                        tp.data.mul_(1 - tau).add_(tau * p.data)
                    for tp, p in zip(target_critic_2.parameters(), critic_2.parameters()):
                        tp.data.mul_(1 - tau).add_(tau * p.data)

            if done:
                last_truncated = bool(truncated)
                break

        # 집계/로그
        if episode_success:
            status = "성공"; success_count += 1
        else:
            status = "실패(충돌)" if last_reason == "collision" else ("실패(시간초과)" if last_truncated else "실패")

        recent_success.append(1 if episode_success else 0)

        # --- 성공률 기반 target_entropy 보정 (B안) ---
        if len(recent_success) >= 20:
            rec_frac = (sum(recent_success) / recent_success.maxlen) if (recent_success.maxlen > 0) else 0.0
            delta = te_lr * (rec_frac - success_target)
            target_entropy = float(np.clip(target_entropy + delta, te_min, te_max))

        success_rate = 100.0 * (success_count / float(ep + 1))
        recent_rate = 100.0 * (sum(recent_success) / max(1, recent_success.maxlen))

        a_print = current_alpha_value()
        print(f"[Episode {ep + 1}] {status} | Return: {total_reward:.2f} | "
              f"누적성공률: {success_rate:.1f}% | 최근100: {recent_rate:.1f}% | alpha={a_print:.4f}")

        # 성공률 임계치 넘으면 α 고정
        if (freeze_alpha_success is not None) and (len(recent_success) == recent_success.maxlen) and auto_alpha and not alpha_frozen:
            sr = sum(recent_success) / len(recent_success)
            if sr >= float(freeze_alpha_success):
                fixed_alpha = a_print
                alpha_frozen = True
                auto_alpha = False
                print(f"[α-freeze] success_rate={sr:.2f}  fixed_alpha={fixed_alpha:.4f}")

        # --- Early Stop: 최근 성공률이 임계치 이상을 연속으로 만족하면 종료 ---
        if (early_stop_success is not None
            and (ep + 1) >= int(early_stop_min_episodes)
            and len(recent_success) == recent_success.maxlen):
            rec_frac = sum(recent_success) / recent_success.maxlen  # 0~1
            if rec_frac >= float(early_stop_success):
                _early_stop_streak += 1
            else:
                _early_stop_streak = 0

            if _early_stop_streak >= int(early_stop_patience):
                print(f"[EarlyStop] 최근{recent_success.maxlen} 성공률 {rec_frac*100:.1f}% "
                      f"≥ {100*float(early_stop_success):.0f}% 를 "
                      f"{early_stop_patience}회 연속 달성 → 학습 종료.")
                break

    print("Training Complete")

    return {
        "actor": actor,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
        "actor_opt": actor_opt,
        "critic_1_opt": critic_1_opt,
        "critic_2_opt": critic_2_opt,
        "replay_buffer": buffer,
        # α 상태 반환
        "log_alpha": log_alpha,
        "alpha_opt": alpha_opt,
        "target_entropy": target_entropy,
        "alpha_mode": ("auto" if (auto_alpha and not alpha_frozen) else "fixed"),
        "fixed_alpha": fixed_alpha,
    }
