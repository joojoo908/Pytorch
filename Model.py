# Model.py — SAC + Alpha control (auto/fixed/clamp/freeze) + BC utils + CKPT compat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Actor (Gaussian policy)
# =====================
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization
        y_t = torch.tanh(x_t)
        action = y_t  # [-1,1]^2  (ENV 해석: angle, speed)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

# =============
# Critic (Twin)
# =============
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        if action.dim() == 3:
            action = action.squeeze(1)
        return self.fc(torch.cat([state, action], dim=1))

# =================
# Replay Buffer (+ckpt)
# =================
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def push(self, *args):
        self.buffer.append(tuple(args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    # ---- serialization ----
    def state_dict(self):
        if len(self.buffer) == 0:
            return {
                "maxlen": self.buffer.maxlen,
                "length": 0,
                "states": None, "actions": None, "rewards": None,
                "next_states": None, "dones": None,
            }
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return {
            "maxlen": self.buffer.maxlen,
            "length": len(self.buffer),
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states),
            "dones": np.array(dones, dtype=np.float32),
        }

    def load_state_dict(self, d):
        self.buffer = deque(maxlen=int(d.get("maxlen", 100000)))
        length = int(d.get("length", 0))
        if length == 0 or d.get("states") is None:
            return
        states = d["states"]
        actions = d["actions"]
        rewards = d["rewards"]
        next_states = d["next_states"]
        dones = d["dones"]
        for i in range(length):
            self.buffer.append((states[i], actions[i], float(rewards[i]), next_states[i], bool(dones[i])))

# =====================
# Checkpoint helpers
# =====================

def save_sac_checkpoint(path,
                        actor, critic_1, critic_2,
                        target_critic_1, target_critic_2,
                        actor_opt, critic_1_opt, critic_2_opt,
                        replay_buffer=None,
                        # α 상태 (옵션)
                        log_alpha=None,
                        alpha_opt_state=None,
                        target_entropy=None,
                        alpha_mode=None,           # 'auto' | 'fixed'
                        fixed_alpha=None):
    ckpt = {
        "actor": actor.state_dict(),
        "critic_1": critic_1.state_dict(),
        "critic_2": critic_2.state_dict(),
        "target_critic_1": target_critic_1.state_dict(),
        "target_critic_2": target_critic_2.state_dict(),
        "actor_opt": actor_opt.state_dict(),
        "critic_1_opt": critic_1_opt.state_dict(),
        "critic_2_opt": critic_2_opt.state_dict(),
    }
    if replay_buffer is not None:
        ckpt["replay_buffer"] = replay_buffer.state_dict()

    # α 관련 (존재 시에만 저장 → 하위호환)
    if log_alpha is not None:
        try:
            val = float(getattr(log_alpha, "detach", lambda: log_alpha)().cpu().item())
        except Exception:
            try:
                val = float(log_alpha)
            except Exception:
                val = None
        if val is not None:
            ckpt["log_alpha"] = val
    if alpha_opt_state is not None:
        ckpt["alpha_opt"] = alpha_opt_state
    if target_entropy is not None:
        try:
            ckpt["target_entropy"] = float(target_entropy)
        except Exception:
            pass
    if alpha_mode is not None:
        ckpt["alpha_mode"] = str(alpha_mode)
    if fixed_alpha is not None:
        ckpt["fixed_alpha"] = float(fixed_alpha)

    torch.save(ckpt, path)


def load_sac_checkpoint(path, state_dim, action_dim):
    actor = GaussianPolicy(state_dim, action_dim).to(device)
    critic_1 = QNetwork(state_dim, action_dim).to(device)
    critic_2 = QNetwork(state_dim, action_dim).to(device)
    target_critic_1 = QNetwork(state_dim, action_dim).to(device)
    target_critic_2 = QNetwork(state_dim, action_dim).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_1_opt = optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_opt = optim.Adam(critic_2.parameters(), lr=3e-4)

    ckpt = torch.load(path, map_location=device, weights_only=False)

    actor.load_state_dict(ckpt["actor"])
    critic_1.load_state_dict(ckpt["critic_1"])
    critic_2.load_state_dict(ckpt["critic_2"])
    target_critic_1.load_state_dict(ckpt["target_critic_1"])
    target_critic_2.load_state_dict(ckpt["target_critic_2"])

    actor_opt.load_state_dict(ckpt["actor_opt"])
    critic_1_opt.load_state_dict(ckpt["critic_1_opt"])
    critic_2_opt.load_state_dict(ckpt["critic_2_opt"])

    replay_buffer = ReplayBuffer(size=ckpt.get("replay_buffer", {}).get("maxlen", 100000))
    if "replay_buffer" in ckpt:
        replay_buffer.load_state_dict(ckpt["replay_buffer"])

    # α 복원(없으면 기본)
    if "log_alpha" in ckpt:
        log_alpha = torch.tensor(float(ckpt["log_alpha"]), device=device, requires_grad=True)
    else:
        log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    if "alpha_opt" in ckpt:
        try:
            alpha_opt.load_state_dict(ckpt["alpha_opt"])
        except Exception:
            pass
    target_entropy = ckpt.get("target_entropy", -float(action_dim))

    # 선택적 모드/값
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
        "replay_buffer": replay_buffer,
        # α 상태 반환
        "log_alpha": log_alpha,
        "alpha_opt": alpha_opt,
        "target_entropy": target_entropy,
        "alpha_mode": alpha_mode,
        "fixed_alpha": fixed_alpha,
    }

# =====================
# SAC Train (with α controls)
# =====================
def sac_train(env,
              actor=None,
              critic_1=None, critic_2=None,
              target_critic_1=None, target_critic_2=None,
              actor_opt=None, critic_1_opt=None, critic_2_opt=None,
              replay_buffer=None,
              episodes=500, batch_size=64, gamma=0.99, tau=0.005,
              # α 자동튜닝 상태 주입/생성
              log_alpha=None, alpha_opt=None, target_entropy=None,
              # α 제어 옵션
              auto_alpha=True, fixed_alpha=None,
              alpha_min=0.03, alpha_max=0.30,
              freeze_alpha_success=None,
              # ▼▼ B안: 성공률→target_entropy 제어 옵션 추가 ▼▼
              success_target=0.90,  # 목표 성공률(0~1)
              te_lr=0.30,  # target_entropy 보정 속도(작게 시작)
              te_min=None,  # 하한(기본: -2*action_dim)
              te_max=None):  # 상한(기본: 0.0)

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

    # 성공률 기반 α 고정
    recent_success = deque(maxlen=100)
    alpha_frozen = False

    te_min = -3.0 * float(action_dim) if te_min is None else float(te_min)
    te_max = -0.1 * float(action_dim) if te_max is None else float(te_max)
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

    for ep in range(episodes):
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
            done = terminated or truncated

            # 종료 원인 파악
            if isinstance(info, dict):
                terms = info.get("reward_terms") or {}
                if terms.get("success", 0):
                    episode_success = True
                    last_reason = "success"
                elif terms.get("collision_reset", False):
                    last_reason = "collision"

            buffer.push(state.cpu().numpy(), action_np, reward, next_state, done)

            state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
            total_reward += float(reward)

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                # 현재 α 값 (모드에 따라)
                a_val = current_alpha_value()
                alpha_tensor = torch.as_tensor(a_val, device=device)

                # ---- Critic update ----
                with torch.no_grad():
                    next_action, log_prob_next = actor.sample(next_states)
                    target_q1 = target_critic_1(next_states, next_action)
                    target_q2 = target_critic_2(next_states, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha_tensor * log_prob_next
                    target_value = rewards + gamma * (1 - dones) * target_q

                q1 = critic_1(states, actions)
                q2 = critic_2(states, actions)
                critic_1_loss = F.mse_loss(q1, target_value)
                critic_2_loss = F.mse_loss(q2, target_value)

                critic_1_opt.zero_grad(); critic_1_loss.backward(); critic_1_opt.step()
                critic_2_opt.zero_grad(); critic_2_loss.backward(); critic_2_opt.step()

                # ---- Actor update ----
                new_action, log_prob = actor.sample(states)
                q1_new = critic_1(states, new_action)
                q2_new = critic_2(states, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha_tensor * log_prob - q_new).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                # ---- α update (auto 모드일 때만) ----
                if auto_alpha and not alpha_frozen:
                    # J(α) = E[ α * (-logπ - H_target) ]
                    alpha_loss = (log_alpha.exp() * (-log_prob - target_entropy).detach()).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                # ---- Polyak ----
                for tp, p in zip(target_critic_1.parameters(), critic_1.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
                for tp, p in zip(target_critic_2.parameters(), critic_2.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

            if done:
                last_truncated = bool(truncated)
                break

        # 집계/로그
        if episode_success:
            status = "성공"; success_count += 1
        else:
            status = "실패(충돌)" if last_reason == "collision" else ("실패(시간초과)" if last_truncated else "실패")

        recent_success.append(1 if episode_success else 0)
        # --- 성공률 기반 target_entropy 보정 ---  <-- [추가]
        if auto_alpha and not alpha_frozen and (len(recent_success) == recent_success.maxlen):
            rec_frac = sum(recent_success) / len(recent_success)  # 0~1, 분모는 len(...)
            delta = te_lr * (success_target - rec_frac)  # 부호 수정!
            target_entropy = float(np.clip(target_entropy + delta, te_min, te_max))

        success_rate = 100.0 * (success_count / float(ep + 1))
        recent_rate = 100.0 * (sum(recent_success) / recent_success.maxlen)

        a_print = current_alpha_value()
        print(f"[Episode {ep + 1}] {status} | Return: {total_reward:.2f} | 누적성공률: {success_rate:.1f}% | 최근100: {recent_rate:.1f}% | alpha={a_print:.4f}")

        # 성공률 임계치 넘으면 α 고정
        if (freeze_alpha_success is not None) and (len(recent_success) == recent_success.maxlen) and auto_alpha and not alpha_frozen:
            sr = sum(recent_success) / len(recent_success)
            if sr >= float(freeze_alpha_success):
                fixed_alpha = a_print
                alpha_frozen = True
                auto_alpha = False
                print(f"[α-freeze] success_rate={sr:.2f}  fixed_alpha={fixed_alpha:.4f}")

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

# =====================
# BC (Behavior Cloning)
# =====================

def expert_action_geodesic(env, speed_gain=0.8):
    """지오데식 맵을 따라가는 간단 전문가 행동을 [-1,1]^2로 생성."""
    geo = getattr(env, "_geo_map", None)
    if geo is None or not np.isfinite(geo).any():
        vec = env.goal_pos - env.agent_pos
    else:
        r, c = env._pos_to_geo_rc(env.agent_pos)
        rows, cols = geo.shape
        best = (float(geo[r, c]) if np.isfinite(geo[r, c]) else np.inf, None)
        N8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in N8:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = float(geo[nr, nc])
                if np.isfinite(v) and v + 1e-6 < best[0]:
                    best = (v, (nr, nc))
        if best[1] is None:
            vec = env.goal_pos - env.agent_pos
        else:
            nxt = env._geo_rc_to_world_center(best[1][0], best[1][1])
            vec = nxt - env.agent_pos

    ang = np.arctan2(vec[1], vec[0])
    dist = np.linalg.norm(vec)
    speed_norm = np.clip(speed_gain * dist / max(getattr(env, "step_size", 1.0), 1e-6), 0.0, 1.0)
    a0 = np.clip(ang / np.pi, -1.0, 1.0)  # angle in [-1,1]
    a1 = 2.0 * speed_norm - 1.0           # speed in [-1,1]
    return np.array([a0, a1], np.float32)


def collect_bc_dataset(env, episodes=300, noise_std=0.05, max_steps=None):
    """전문가 정책으로 (obs, action) 쌍 수집."""
    Xs, Ys = [], []
    steps_limit = max_steps or getattr(env, "max_steps", 300)
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(steps_limit):
            a = expert_action_geodesic(env)
            Xs.append(obs.astype(np.float32))
            if noise_std > 0.0:
                a = np.clip(a + np.random.normal(0, noise_std, size=a.shape), -1.0, 1.0)
            Ys.append(a.astype(np.float32))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.stack(Xs), np.stack(Ys)


def bc_pretrain_actor(env, actor, dataset=None, epochs=10, batch_size=256, lr=3e-4):
    """actor의 mean을 tanh로 스케일해 데모 액션을 MSE로 모방."""
    if dataset is None:
        X, Y = collect_bc_dataset(env, episodes=300, noise_std=0.05)
    else:
        X, Y = dataset
    actor.train()
    opt = optim.Adam(actor.parameters(), lr=lr)
    N = X.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.from_numpy(X[idx]).float().to(device)
            yb = torch.from_numpy(Y[idx]).float().to(device)
            mean, _ = actor.forward(xb)
            pred = torch.tanh(mean)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return actor
