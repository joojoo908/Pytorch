import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


class CategoricalPolicyPPO(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.forward(x))

    def sample(self, x):
        dist = self.dist(x)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logp, entropy

    def evaluate_actions(self, x, action):
        dist = self.dist(x)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return logp, entropy


class TargetedCategoricalPolicyPPO(nn.Module):
    def __init__(self, obs_dim: int, target_dim: int, choice_dim: int, hidden: int = 128):
        super().__init__()
        self.target_dim = target_dim
        self.choice_dim = choice_dim
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.target_logits = nn.Linear(hidden, target_dim)
        self.choice_logits = nn.Linear(hidden, choice_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.target_logits(x), self.choice_logits(x)

    def dists(self, x, target_mask=None):
        target_logits, choice_logits = self.forward(x)
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask.bool(), -1e9)
        return (
            torch.distributions.Categorical(logits=target_logits),
            torch.distributions.Categorical(logits=choice_logits),
        )

    def sample(self, x, target_mask=None):
        target_dist, choice_dist = self.dists(x, target_mask)
        target = target_dist.sample()
        choice = choice_dist.sample()
        logp = target_dist.log_prob(target) + choice_dist.log_prob(choice)
        entropy = target_dist.entropy() + choice_dist.entropy()
        return target, choice, logp, entropy

    def evaluate_actions(self, x, target, choice, target_mask=None):
        target_dist, choice_dist = self.dists(x, target_mask)
        logp = target_dist.log_prob(target) + choice_dist.log_prob(choice)
        entropy = target_dist.entropy() + choice_dist.entropy()
        return logp, entropy


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x).squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    steps = len(rewards)
    adv = np.zeros(steps, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv, returns
