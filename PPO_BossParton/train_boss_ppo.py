import os
import argparse
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import torch

from boss_pattern_env import BossEnvConfig, BossPatternEnv
from ppo_model import TargetedCategoricalPolicyPPO, ValueNetwork, compute_gae, to_tensor


@dataclass
class TrainConfig:
    total_updates: int = 400
    rollout_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 8
    batch_size: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.02
    conditional_balance_coef: float = 1.00
    conditional_uniform_kl_coef: float = 0.20
    conditional_skill_floor: float = 0.10
    conditional_skill_cap: float = 0.55
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    save_every: int = 50
    checkpoint_dir: str = "checkpoints_targeted"
    seed: int = 7
    boss_kind: str = "brass"


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.target = []
        self.choice = []
        self.logp = []
        self.reward = []
        self.done = []
        self.value = []
        self.target_mask = []

    def add(self, obs, target, choice, logp, reward, done, value, target_mask):
        self.obs.append(obs)
        self.target.append(target)
        self.choice.append(choice)
        self.logp.append(logp)
        self.reward.append(reward)
        self.done.append(done)
        self.value.append(value)
        self.target_mask.append(target_mask)

    def clear(self):
        self.__init__()

    def size(self):
        return len(self.reward)


def collect_rollout(env, actor, critic, device, rollout_steps):
    obs, _ = env.reset()
    buffer = RolloutBuffer()
    episode_reward = 0.0
    episode_count = 0
    episodic_rewards = []

    for _ in range(rollout_steps):
        obs_t = to_tensor(obs, device).unsqueeze(0)
        target_mask = np.asarray(
            [obs[5 + idx * 5 + 4] > 0.5 for idx in range(env.target_dim)],
            dtype=np.bool_,
        )
        target_mask_t = torch.as_tensor(target_mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            value = float(critic(obs_t).item())
            target_t, choice_t, logp_t, _ = actor.sample(obs_t, target_mask_t)

        target = int(target_t.item())
        choice = int(choice_t.item())
        action = (target, choice)
        next_obs, reward, done, trunc, info = env.step(action)
        terminal = bool(done or trunc)

        buffer.add(
            obs.astype(np.float32),
            target,
            choice,
            float(logp_t.item()),
            float(reward),
            1.0 if terminal else 0.0,
            value,
            target_mask,
        )

        episode_reward += reward
        obs = next_obs

        if terminal:
            episodic_rewards.append(episode_reward)
            episode_reward = 0.0
            episode_count += 1
            obs, _ = env.reset()

    obs_t = to_tensor(obs, device).unsqueeze(0)
    with torch.no_grad():
        last_value = float(critic(obs_t).item())

    return buffer, last_value, episodic_rewards, episode_count


def ppo_update(buffer, bootstrap_value, actor, critic, actor_opt, critic_opt, device, cfg):
    obs = np.asarray(buffer.obs, dtype=np.float32)
    target = np.asarray(buffer.target, dtype=np.int64)
    choice = np.asarray(buffer.choice, dtype=np.int64)
    old_logp = np.asarray(buffer.logp, dtype=np.float32)
    reward = np.asarray(buffer.reward, dtype=np.float32)
    done = np.asarray(buffer.done, dtype=np.float32)
    value = np.asarray(buffer.value, dtype=np.float32)
    target_mask = np.asarray(buffer.target_mask, dtype=np.bool_)

    values_ext = np.concatenate([value, np.array([bootstrap_value], dtype=np.float32)], axis=0)
    adv, ret = compute_gae(reward, values_ext, done, gamma=cfg.gamma, lam=cfg.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs_t = to_tensor(obs, device)
    target_t = torch.as_tensor(target, dtype=torch.long, device=device)
    choice_t = torch.as_tensor(choice, dtype=torch.long, device=device)
    old_logp_t = to_tensor(old_logp, device)
    adv_t = to_tensor(adv, device)
    ret_t = to_tensor(ret, device)
    target_mask_t = torch.as_tensor(target_mask, dtype=torch.bool, device=device)

    indices = np.arange(buffer.size())
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    update_count = 0

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, buffer.size(), cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            b_obs = obs_t[batch_idx]
            b_target = target_t[batch_idx]
            b_choice = choice_t[batch_idx]
            b_old_logp = old_logp_t[batch_idx]
            b_adv = adv_t[batch_idx]
            b_ret = ret_t[batch_idx]
            b_target_mask = target_mask_t[batch_idx]

            new_logp, entropy = actor.evaluate_actions(
                b_obs, b_target, b_choice, b_target_mask
            )
            ratio = torch.exp(new_logp - b_old_logp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
            actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy.mean()

            _, choice_logits = actor(b_obs)
            skill_probs = torch.softmax(choice_logits[:, 1:], dim=-1)
            alive_mask_id = (
                (b_obs[:, 9] > 0.5).long()
                + 2 * (b_obs[:, 14] > 0.5).long()
                + 4 * (b_obs[:, 19] > 0.5).long()
            )
            situation_id = alive_mask_id * actor.target_dim + b_target
            conditional_penalties = []
            for bucket in torch.unique(situation_id):
                in_bucket = situation_id == bucket
                if int(in_bucket.sum().item()) < 2:
                    continue
                mean_skill_probs = skill_probs[in_bucket].mean(dim=0)
                floor_penalty = torch.relu(
                    cfg.conditional_skill_floor - mean_skill_probs
                ).pow(2).sum()
                cap_penalty = torch.relu(
                    mean_skill_probs - cfg.conditional_skill_cap
                ).pow(2).sum()
                # Reverse KL from a uniform four-skill prior keeps gradients
                # strong even when one skill's probability has nearly collapsed.
                uniform_kl = -torch.log(
                    mean_skill_probs.clamp_min(1e-6)
                ).mean()
                conditional_penalties.append(
                    cfg.conditional_balance_coef * (floor_penalty + cap_penalty)
                    + cfg.conditional_uniform_kl_coef * uniform_kl
                )
            if conditional_penalties:
                actor_loss = actor_loss + torch.stack(conditional_penalties).mean()

            value_pred = critic(b_obs)
            critic_loss = cfg.value_coef * (b_ret - value_pred).pow(2).mean()

            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            actor_loss_sum += float(actor_loss.item())
            critic_loss_sum += float(critic_loss.item())
            update_count += 1

    return actor_loss_sum / max(1, update_count), critic_loss_sum / max(1, update_count)


def save_checkpoint(path, actor, critic, train_cfg, env_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "train_config": asdict(train_cfg),
            "env_config": asdict(env_cfg),
        },
        path,
    )


def run_training(train_cfg: TrainConfig):
    env_cfg = BossEnvConfig(
        seed=train_cfg.seed,
        boss_kind=train_cfg.boss_kind,
        randomize_party_composition=True,
    )

    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BossPatternEnv(env_cfg)

    actor = TargetedCategoricalPolicyPPO(
        env.obs_dim, env.target_dim, env.choice_dim
    ).to(device)
    critic = ValueNetwork(env.obs_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=train_cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=train_cfg.critic_lr)

    recent_rewards = []
    for update in range(1, train_cfg.total_updates + 1):
        rollout, bootstrap_value, episode_rewards, episode_count = collect_rollout(
            env, actor, critic, device, train_cfg.rollout_steps
        )
        actor_loss, critic_loss = ppo_update(
            rollout, bootstrap_value, actor, critic, actor_opt, critic_opt, device, train_cfg
        )

        if episode_rewards:
            recent_rewards.extend(episode_rewards)
            recent_rewards = recent_rewards[-20:]
        avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0

        print(
            f"[{train_cfg.boss_kind}][update {update:04d}] "
            f"steps={rollout.size():4d} "
            f"episodes={episode_count:2d} "
            f"avg_reward={avg_reward:7.3f} "
            f"actor_loss={actor_loss:7.4f} "
            f"critic_loss={critic_loss:7.4f}"
        )

        if update % train_cfg.save_every == 0 or update == train_cfg.total_updates:
            ckpt_name = f"{train_cfg.boss_kind}_targeted_ppo_{update:04d}.pth"
            save_checkpoint(
                os.path.join(train_cfg.checkpoint_dir, ckpt_name),
                actor,
                critic,
                train_cfg,
                env_cfg,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", choices=["brass", "dragon", "both"], default="both")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="checkpoints_targeted")
    args = parser.parse_args()

    if args.boss == "both":
        processes = []
        for offset, boss in enumerate(("brass", "dragon")):
            cfg = TrainConfig(
                boss_kind=boss,
                seed=7 + offset,
                total_updates=args.updates,
                rollout_steps=args.rollout_steps,
                save_every=args.save_every,
                checkpoint_dir=args.checkpoint_dir,
            )
            process = mp.Process(target=run_training, args=(cfg,), name=f"train_{boss}")
            process.start()
            processes.append(process)

        exit_code = 0
        for process in processes:
            process.join()
            if process.exitcode != 0:
                exit_code = process.exitcode or 1
        raise SystemExit(exit_code)

    run_training(
        TrainConfig(
            boss_kind=args.boss,
            total_updates=args.updates,
            rollout_steps=args.rollout_steps,
            save_every=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
        )
    )


if __name__ == "__main__":
    main()
