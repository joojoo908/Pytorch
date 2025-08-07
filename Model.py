# sac_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor (확률적 정책)
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
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
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

# Critic
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        if action.dim() == 3:
            action = action.squeeze(1)
        return self.fc(torch.cat([state, action], dim=1))

# Replay Buffer
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
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# SAC 학습 함수
def sac_train(env, actor=None, critic_1=None, critic_2=None,
              target_critic_1=None, target_critic_2=None,
              episodes=500, batch_size=64, gamma=0.99, tau=0.005):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 모델 초기화 또는 외부에서 주입받은 모델 사용
    if actor is None:
        actor = GaussianPolicy(state_dim, action_dim).to(device)
    else:
        actor = actor.to(device)

    if critic_1 is None:
        critic_1 = QNetwork(state_dim, action_dim).to(device)
    else:
        critic_1 = critic_1.to(device)

    if critic_2 is None:
        critic_2 = QNetwork(state_dim, action_dim).to(device)
    else:
        critic_2 = critic_2.to(device)

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

    # 옵티마이저
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_1_opt = optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_opt = optim.Adam(critic_2.parameters(), lr=3e-4)

    buffer = ReplayBuffer()
    alpha = 0.2  # entropy coefficient

    for ep in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(np.array(state)).to(device)
        total_reward = 0

        for t in range(env.max_steps):
            with torch.no_grad():
                action, _ = actor.sample(state.unsqueeze(0))
            action_np = action.cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.push(state.cpu().numpy(), action_np, reward, next_state, done)

            state = torch.FloatTensor(np.array(next_state)).to(device)
            total_reward += reward

            if len(buffer) < batch_size:
                continue

            # 샘플링 및 업데이트 코드 (동일)
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            with torch.no_grad():
                next_action, log_prob = actor.sample(next_states)
                target_q1 = target_critic_1(next_states, next_action)
                target_q2 = target_critic_2(next_states, next_action)
                target_q = torch.min(target_q1, target_q2) - alpha * log_prob
                target_value = rewards + gamma * (1 - dones) * target_q

            q1 = critic_1(states, actions)
            q2 = critic_2(states, actions)
            critic_1_loss = F.mse_loss(q1, target_value)
            critic_2_loss = F.mse_loss(q2, target_value)

            critic_1_opt.zero_grad()
            critic_1_loss.backward()
            critic_1_opt.step()

            critic_2_opt.zero_grad()
            critic_2_loss.backward()
            critic_2_opt.step()

            new_action, log_prob = actor.sample(states)
            q1_new = critic_1(states, new_action)
            q2_new = critic_2(states, new_action)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (alpha * log_prob - q_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_critic_2.parameters(), critic_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break

        print(f"[Episode {ep + 1}] Total Reward: {total_reward:.2f}")

    print("Training Complete")
    return actor
