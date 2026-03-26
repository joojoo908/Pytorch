# ppo_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device)

class GaussianPolicyPPO(nn.Module):
    """
    연속 액션용 PPO Actor:
    - mean = MLP(state)
    - log_std = learnable parameter (state-independent)
    - action = tanh(z) 로 squash ([-1,1])
    - log_prob는 squash 보정 포함
    """
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_init=-0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, act_dim)

        self.log_std = nn.Parameter(torch.ones(act_dim) * float(log_std_init))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mean(x)
        return mu

    def _dist(self, x):
        mu = self.forward(x)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def sample(self, x):
        """
        x: (B,obs)
        return:
          action: (B,act) in [-1,1]
          logp: (B,)
          entropy: (B,)
        """
        dist = self._dist(x)
        z = dist.rsample()
        a = torch.tanh(z)

        # squash 보정: logp(a) = logp(z) - sum(log(1 - tanh(z)^2))
        logp_z = dist.log_prob(z).sum(dim=-1)
        correction = torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        logp = logp_z - correction

        entropy = dist.entropy().sum(dim=-1)
        return a, logp, entropy

    def evaluate_actions(self, x, a):
        """
        PPO 업데이트용: 주어진 action a의 log_prob, entropy 계산
        a는 [-1,1] 범위(tanh된 값)라고 가정
        """
        dist = self._dist(x)

        # atanh로 z 복원 (클램프 필수)
        a_clamped = torch.clamp(a, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + a_clamped) / (1 - a_clamped))

        logp_z = dist.log_prob(z).sum(dim=-1)
        correction = torch.log(1.0 - a_clamped.pow(2) + 1e-6).sum(dim=-1)
        logp = logp_z - correction

        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x).squeeze(-1)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T,)
    values: (T+1,)  bootstrap 포함
    dones:  (T,)    done이면 1.0
    return:
      adv: (T,)
      ret: (T,)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret
