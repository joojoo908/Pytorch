import json
import os

import numpy as np
import torch

from boss_pattern_env import BossEnvConfig, BossPatternEnv
from ppo_model import TargetedCategoricalPolicyPPO


CHECKPOINT_DIR = "checkpoints_conditional_20d"
SAMPLES_PER_MASK = 4000


def load_actor(boss_kind):
    path = os.path.join(
        CHECKPOINT_DIR, f"{boss_kind}_targeted_ppo_0400.pth"
    )
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["actor"]
    obs_dim = int(state["fc1.weight"].shape[1])
    target_dim = int(state["target_logits.weight"].shape[0])
    choice_dim = int(state["choice_logits.weight"].shape[0])
    actor = TargetedCategoricalPolicyPPO(obs_dim, target_dim, choice_dim)
    actor.load_state_dict(state)
    actor.eval()
    finite = all(torch.isfinite(value).all().item() for value in state.values())
    return actor, obs_dim, target_dim, choice_dim, finite


def make_observations(boss_kind, alive_mask, count, seed):
    env = BossPatternEnv(
        BossEnvConfig(
            boss_kind=boss_kind,
            seed=seed,
            randomize_party_composition=False,
        )
    )
    rng = np.random.RandomState(seed)
    observations = []
    for _ in range(count):
        env.reset()
        env.player_alive[:] = alive_mask
        env.player_hp[:] = 0.0
        for idx, alive in enumerate(alive_mask):
            if alive:
                env.player_hp[idx] = env.player_max_hp[idx] * rng.uniform(0.10, 1.0)
        env.boss_hp = env.cfg.boss_max_hp * rng.uniform(0.10, 1.0)
        env.step_count = int(rng.randint(0, env.cfg.max_steps))
        env.recent_damage_taken = float(rng.uniform(0.0, 180.0))
        env.last_skill_choice = int(rng.randint(0, 5))
        env.last_target = int(rng.randint(0, 3))
        env.boss_pos[:] = rng.uniform(-1200.0, 1200.0, size=2)
        env.player_pos[:] = rng.uniform(-1500.0, 1500.0, size=(3, 2))
        observations.append(env._build_obs())
    return np.asarray(observations, dtype=np.float32)


def normalized(values):
    total = float(np.sum(values))
    if total <= 0:
        return [0.0 for _ in values]
    return [round(float(value / total), 4) for value in values]


def evaluate_boss(boss_kind):
    actor, obs_dim, target_dim, choice_dim, finite = load_actor(boss_kind)
    all_results = {}
    target_conditioned = {
        role: np.zeros(choice_dim, dtype=np.float64)
        for role in ("bass", "drum", "guitar")
    }
    target_conditioned_counts = {role: 0 for role in target_conditioned}
    role_names = ("bass", "drum", "guitar")

    for bits in range(1, 8):
        alive_mask = np.asarray(
            [(bits & (1 << idx)) != 0 for idx in range(3)], dtype=np.bool_
        )
        obs = make_observations(
            boss_kind, alive_mask, SAMPLES_PER_MASK, seed=1000 + bits
        )
        with torch.no_grad():
            target_logits, choice_logits = actor(torch.from_numpy(obs))
            masked_targets = target_logits.masked_fill(
                ~torch.from_numpy(np.repeat(alive_mask[None, :], len(obs), axis=0)),
                -1e9,
            )
            targets = torch.argmax(masked_targets, dim=1).numpy()
            raw_choices = torch.argmax(choice_logits, dim=1).numpy()
            choice_probs = torch.softmax(choice_logits, dim=1).mean(dim=0).numpy()

        server_counts = np.zeros(choice_dim, dtype=np.float64)
        logits_np = choice_logits.numpy().copy()
        last_choices = np.rint(obs[:, 3] * 4.0).astype(np.int64)
        for row, last_choice in zip(logits_np, last_choices):
            adjusted = row.copy()
            if 1 <= last_choice <= 4:
                adjusted[last_choice] = -np.inf
            order = np.argsort(adjusted)
            server_counts[order[-1]] += 0.75
            server_counts[order[-2]] += 0.25

        raw_counts = np.bincount(raw_choices, minlength=choice_dim)
        target_counts = np.bincount(targets, minlength=target_dim)
        mask_name = "".join(
            role_names[idx][0].upper()
            for idx, alive in enumerate(alive_mask)
            if alive
        )
        all_results[mask_name] = {
            "alive": [
                role_names[idx] for idx, alive in enumerate(alive_mask) if alive
            ],
            "target_argmax": normalized(target_counts),
            "choice_raw_argmax": normalized(raw_counts),
            "choice_server_expected": normalized(server_counts),
            "choice_mean_probability": [
                round(float(value), 4) for value in choice_probs
            ],
        }
        for idx, target in enumerate(targets):
            role = role_names[int(target)]
            adjusted = logits_np[idx].copy()
            last_choice = last_choices[idx]
            if 1 <= last_choice <= 4:
                adjusted[last_choice] = -np.inf
            order = np.argsort(adjusted)
            target_conditioned[role][order[-1]] += 0.75
            target_conditioned[role][order[-2]] += 0.25
            target_conditioned_counts[role] += 1

    return {
        "checkpoint": f"{boss_kind}_targeted_ppo_0400.pth",
        "obs_dim": obs_dim,
        "target_dim": target_dim,
        "choice_dim": choice_dim,
        "weights_finite": finite,
        "labels": ["move", "skill1", "skill2", "skill3", "skill4"],
        "by_alive_party": all_results,
        "by_selected_target_server_expected": {
            role: normalized(target_conditioned[role])
            for role in target_conditioned
            if target_conditioned_counts[role] > 0
        },
    }


def main():
    result = {
        boss_kind: evaluate_boss(boss_kind)
        for boss_kind in ("brass", "dragon")
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
