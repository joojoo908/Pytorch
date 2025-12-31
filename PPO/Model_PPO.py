# Model_PPO.py
# -----------------------------------------
# [PPO 전용 학습 코드]
# - 기존 SAC 코드(Model.py)는 그대로 두고,
#   비교 실험을 위해 PPO 버전만 별도 파일로 분리했습니다.
# -----------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Env API compatibility (Gym / Gymnasium)
# ----------------------------

def reset_env(env):
    """Gym(gymnasium) reset()의 (obs) / (obs, info) 둘 다 지원."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out

def step_env(env, action):
    """단일 (obs, reward, done, info) 형식으로 정규화."""
    out = env.step(action)
    # gymnasium: (obs, r, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, r, done, info
    # gym: (obs, r, done, info)
    elif isinstance(out, tuple) and len(out) == 4:
        return out
    raise RuntimeError("Unsupported env.step(...) return format")

# ----------------------------
# Helpers
# ----------------------------

def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)

def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

# ----------------------------
# PPO Networks
# ----------------------------

class GaussianPolicyPPO(nn.Module):
    """
    [변경점/설명]
    - SAC에서 쓰던 tanh-squash 정책 대신,
      PPO용으로 '그냥 가우시안' 정책을 따로 정의했습니다.
    - ENV쪽에서 action을 [-1, 1]로 clip하고 있기 때문에
      여기서는 tanh로 다시 한 번 감지(보정)하지 않습니다.
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: Tuple[int, ...] = (512, 512, 512),
                 log_std_bounds: Tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean, log_std = torch.split(h, self.act_dim, dim=-1)

        # [PPO용 안정화] log_std를 적당한 범위 안에 넣어 줌
        log_std = torch.tanh(log_std)  # [-1, 1]
        log_std = self.log_std_min + 0.5 * (log_std + 1.0) * (self.log_std_max - self.log_std_min)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [중요]
        - 정책으로부터 행동을 샘플링하고,
          그 행동에 대한 log_prob / entropy를 같이 반환합니다.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + std * noise

        # 가우시안 log_prob
        var = std.pow(2)
        log_prob = -0.5 * (((action - mean) ** 2) / (var + 1e-8)
                           + 2.0 * log_std + math.log(2.0 * math.pi))
        log_prob = log_prob.sum(dim=-1)

        # analytic entropy
        entropy = 0.5 + 0.5 * math.log(2.0 * math.pi) + log_std
        entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        평가 모드에서 mean만 사용하는 결정적 정책.
        (테스트할 때 사용 가능)
        """
        mean, _ = self.forward(obs)
        return mean

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [PPO 핵심]
        - 이미 rollout에서 샘플한 행동 a_t에 대해,
          현재 정책 π_θ(a_t | s_t)의 log_prob를 다시 계산.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        var = std.pow(2)

        # log_prob(a_t | s_t; θ)
        log_prob = -0.5 * (((actions - mean) ** 2) / (var + 1e-8)
                           + 2.0 * log_std + math.log(2.0 * math.pi))
        log_prob = log_prob.sum(dim=-1)

        entropy = 0.5 + 0.5 * math.log(2.0 * math.pi) + log_std
        entropy = entropy.sum(dim=-1)

        return log_prob, entropy

class ValueNetwork(nn.Module):
    """
    상태 가치함수 V(s)를 근사하는 Critic 네트워크.
    (SAC의 Q(s,a)와는 다름)
    """
    def __init__(self, obs_dim: int, hidden: Tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = self.net(obs)
        return v.squeeze(-1)

# ----------------------------
# GAE 계산
# ----------------------------

def compute_gae(rewards: np.ndarray,
                values: np.ndarray,
                dones: np.ndarray,
                gamma: float,
                lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    [GAE(lambda)]
    rewards: [T]
    values:  [T+1] (마지막은 bootstrap 값; terminal이면 0)
    dones:   [T]   (True면 그 시점에서 에피소드 종료)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

# ----------------------------
# PPO Training Loop
# ----------------------------

def ppo_train(env,
              actor: Optional[GaussianPolicyPPO] = None,
              critic: Optional[ValueNetwork] = None,
              episodes: int = 1000,
              max_steps: int = 512,
              gamma: float = 0.99,
              lam: float = 0.95,
              clip_eps: float = 0.20,
              actor_lr: float = 3e-4,
              critic_lr: float = 3e-4,
              epochs: int = 10,
              entropy_coef: float = 0.0,
              value_coef: float = 0.5,
              max_grad_norm: float = 0.5,
              device: Optional[torch.device] = None,
              save_best_online: bool = True,
              best_delta: float = 0.02,
              best_min_episodes: int = 30,
              best_actor_path: str = "ppo_actor_best.pth"):
    """
    [변경 요약]
    - 기존 SAC 학습 루프(sac_train)와 달리,
      이 함수는 on-policy PPO 알고리즘을 구현합니다.
    - 한 에피소드에서 trajectory를 모은 뒤,
      GAE로 advantage/return을 계산하고,
      같은 데이터를 여러 epoch 반복 학습합니다.
    """
    assert episodes > 0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 관측/행동 차원 추론
    _probe = reset_env(env)
    if isinstance(_probe, (list, tuple, np.ndarray)):
        obs_dim = int(np.asarray(_probe).shape[-1])
    else:
        obs_dim = int(len(_probe))

    if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        act_dim = int(env.action_space.shape[0])
    else:
        act_dim = 2

    # 네트워크 초기화
    if actor is None:
        actor = GaussianPolicyPPO(obs_dim, act_dim).to(device)
    if critic is None:
        critic = ValueNetwork(obs_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    recent_success: deque[int] = deque(maxlen=100)
    best_score = -1.0

    for ep in range(episodes):
        obs = reset_env(env)
        done = False
        ep_steps = 0
        ep_reward = 0.0

        # 한 에피소드 trajectory 버퍼
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        logprobs: List[float] = []
        rewards: List[float] = []
        dones: List[bool] = []
        values: List[float] = []

        info: Dict[str, Any] = {}
        last_reward = 0.0

        # -------- Rollout 수집 --------
        while (not done) and (ep_steps < max_steps):
            s = to_tensor(obs, device).unsqueeze(0)

            with torch.no_grad():
                v = critic(s).item()
                a, logp, ent = actor.sample(s)

            a_np = a.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = step_env(env, a_np)

            states.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(a_np, dtype=np.float32))
            logprobs.append(float(logp.item()))
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(v))

            obs = next_obs
            ep_reward += float(reward)
            ep_steps += 1
            last_reward = float(reward)

        # -------- GAE / Return 계산 --------
        if done:
            next_v = 0.0
        else:
            with torch.no_grad():
                s_last = to_tensor(obs, device).unsqueeze(0)
                next_v = float(critic(s_last).item())

        values_np = np.asarray(values + [next_v], dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)

        adv_np, ret_np = compute_gae(rewards_np, values_np, dones_np, gamma, lam)

        # 텐서 변환
        states_t = to_tensor(np.stack(states, axis=0), device)
        actions_t = to_tensor(np.stack(actions, axis=0), device)
        old_logp_t = to_tensor(np.asarray(logprobs, dtype=np.float32), device)
        adv_t = to_tensor(adv_np, device)
        ret_t = to_tensor(ret_np, device)

        # advantage 정규화 (학습 안정화에 매우 중요)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # -------- PPO 업데이트 (full batch, 여러 epoch) --------
        n_steps = states_t.shape[0]
        for _ in range(epochs):
            new_logp, entropy = actor.evaluate_actions(states_t, actions_t)
            ratio = (new_logp - old_logp_t).exp()  # π_θ / π_θ_old

            # 클리핑된 surrogate objective
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # value loss (MSE)
            v_pred = critic(states_t)
            value_loss = (ret_t - v_pred).pow(2).mean()

            # 최종 loss = policy + value - entropy
            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            actor_opt.step()
            critic_opt.step()

        # -------- 에피소드 성공 여부 판정 --------
        episode_success = False
        if isinstance(info, dict):
            terms = (info.get("reward_terms") or {})
            if terms.get("success", 0):
                episode_success = True
        if (not episode_success) and isinstance(info, dict) and info.get("success", False):
            episode_success = True
        if (not episode_success) and (float(last_reward) > 100.0):
            episode_success = True
        if (not episode_success) and isinstance(info, dict):
            if "dist_to_goal" in info and info["dist_to_goal"] is not None:
                try:
                    dist_end = float(info["dist_to_goal"])
                    radius = float(getattr(env, "success_radius", 0.5))
                    if dist_end <= radius + 1e-6:
                        episode_success = True
                except Exception:
                    pass

        recent_success.append(1 if episode_success else 0)
        recent_rate = 100.0 * (sum(recent_success) / max(1, len(recent_success)))

        if (ep + 1) % 10 == 0:
            print(
                f"[PPO EP {ep+1:5d}] steps={ep_steps:3d}  R={ep_reward:8.2f}  "
                f"succ={int(episode_success)}  recent@{len(recent_success)}={recent_rate:5.1f}%"
            )

        # -------- online best 저장 --------
        if save_best_online and len(recent_success) >= best_min_episodes:
            recent_mean = sum(recent_success) / len(recent_success)
            if recent_mean >= best_score + best_delta:
                best_score = float(recent_mean)
                torch.save(actor.state_dict(), best_actor_path)
                print(f"[PPO BEST-online] ep={ep+1} mean={recent_mean:.3f} → saved {best_actor_path}")

    # 최종 actor 저장
    torch.save(actor.state_dict(), "ppo_actor_last.pth")
    return {
        "actor": actor,
        "critic": critic,
    }
