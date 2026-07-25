import argparse
import os
import time

from boss_pattern_env import BossEnvConfig, BossPatternEnv


SKILL_NAMES = {
    "brass": {
        0: "Move Only / No Skill",
        1: "Brass Skill 1: Summon Adds",
        2: "Brass Skill 2: Projectile 100",
        3: "Brass Skill 3: 16 Spread Shots",
        4: "Brass Skill 4: Checkerboard A-B-A (4 beats, 50 each)",
    },
    "dragon": {
        0: "Move Only / No Skill",
        1: "Dragon Skill 1: Circular AoE",
        2: "Dragon Skill 2: Heal 100 + Summon",
        3: "Dragon Skill 3: Knockback 600 + Silence",
        4: "Dragon Skill 4: 5 Random Explosions (4 beats, 50 each)",
    },
}


def find_latest_checkpoint(boss_kind: str, checkpoint_dir: str) -> str | None:
    prefix = f"{boss_kind}_targeted_ppo_"
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        name for name in os.listdir(checkpoint_dir)
        if name.startswith(prefix) and name.endswith(".pth")
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(checkpoint_dir, candidates[-1])


def load_actor(checkpoint_path: str, env: BossPatternEnv):
    import torch

    from ppo_model import TargetedCategoricalPolicyPPO

    data = torch.load(checkpoint_path, map_location="cpu")
    actor = TargetedCategoricalPolicyPPO(env.obs_dim, env.target_dim, env.choice_dim)
    actor.load_state_dict(data["actor"])
    actor.eval()
    return actor


def select_action(mode: str, env: BossPatternEnv, obs, actor, step_idx: int, scripted_actions):
    if mode == "manual":
        while True:
            raw = input("target skill (e.g. '2 4', skill 0=move only; Enter=random, q=quit): ").strip().lower()
            if raw == "q":
                raise KeyboardInterrupt
            if raw == "":
                return (
                    int(env.rng.randint(0, env.target_dim)),
                    int(env.rng.randint(0, env.choice_dim)),
                )
            values = raw.replace(",", " ").split()
            if len(values) == 2 and all(value.lstrip("-").isdigit() for value in values):
                target_idx, skill_choice = map(int, values)
                if 0 <= target_idx < env.cfg.n_players and 0 <= skill_choice < env.skill_option_count:
                    return target_idx, skill_choice
            print("invalid input")

    if mode == "random":
        return (
            int(env.rng.randint(0, env.target_dim)),
            int(env.rng.randint(0, env.choice_dim)),
        )

    if mode == "sequence":
        if not scripted_actions:
            raise ValueError("sequence mode requires at least one action")
        return scripted_actions[(step_idx - 1) % len(scripted_actions)]

    import torch

    from ppo_model import to_tensor

    obs_t = to_tensor(obs, "cpu").unsqueeze(0)
    with torch.no_grad():
        target_logits, choice_logits = actor(obs_t)
        if mode == "policy-sample":
            target_dist = torch.distributions.Categorical(logits=target_logits)
            choice_dist = torch.distributions.Categorical(logits=choice_logits)
            return int(target_dist.sample().item()), int(choice_dist.sample().item())
        return (
            int(torch.argmax(target_logits, dim=-1).item()),
            int(torch.argmax(choice_logits, dim=-1).item()),
        )


def format_players(env: BossPatternEnv):
    parts = []
    for idx in range(env.cfg.n_players):
        dist = float(((env.player_pos[idx] - env.boss_pos) ** 2).sum() ** 0.5)
        hp = float(env.player_hp[idx])
        alive = bool(env.player_alive[idx] > 0.5)
        role = env.player_roles[idx]
        threat = float(env.player_threat_score[idx]) if hasattr(env, "player_threat_score") else 0.0
        dps = float(env.player_recent_boss_dps[idx]) if hasattr(env, "player_recent_boss_dps") else 0.0
        heal = float(env.player_recent_heal[idx]) if hasattr(env, "player_recent_heal") else 0.0
        parts.append(
            f"P{idx}/{role}: hp={hp:5.1f} pos=({env.player_pos[idx, 0]:6.0f},{env.player_pos[idx, 1]:6.0f}) "
            f"dist={dist:6.1f} "
            f"dps={dps:4.0f} heal={heal:4.0f} threat={threat:5.1f} alive={'Y' if alive else 'N'}"
        )
    return " | ".join(parts)


def print_step_summary(env: BossPatternEnv, boss_kind: str, step_idx: int, action, reward: float, info, boss_hp_before: float, player_hp_before):
    boss_hp_after = float(env.boss_hp)
    player_hp_after = env.player_hp.copy()
    deltas = player_hp_before - player_hp_after
    hit_indices = [idx for idx, delta in enumerate(deltas) if delta > 1e-6]
    hit_success = bool(hit_indices)

    target_idx = info["target_idx"]
    skill_choice = info["skill_choice"]
    print(
        f"\n[step {step_idx:02d}] action=(target={target_idx}, choice={skill_choice}) "
        f"target=P{target_idx}/{info['target_role']} "
        f"choice={skill_choice} {SKILL_NAMES[boss_kind][skill_choice]}"
    )
    print(
        f" reward={reward:6.3f} "
        f"boss_pos=({env.boss_pos[0]:6.0f},{env.boss_pos[1]:6.0f}) "
        f"moved={info.get('move_distance', 0.0):5.1f} "
        f"boss_hp={boss_hp_before:6.1f}->{boss_hp_after:6.1f} "
        f"boss_damage_dealt={info['boss_damage_dealt']:6.1f} "
        f"(instant={info.get('instant_boss_damage', 0.0):5.1f}, ongoing={info.get('ongoing_boss_damage', 0.0):5.1f}) "
        f"weighted={info.get('weighted_boss_damage', 0.0):6.1f} "
        f"boss_damage_taken={info['boss_damage_taken']:5.1f} "
        f"heal_done={info.get('player_heal_done', 0.0):4.1f}"
    )
    if info.get("skill_note") or info.get("ongoing_note"):
        print(f" note=skill[{info.get('skill_note', '')}] ongoing[{info.get('ongoing_note', '')}]")
    if hit_success:
        print(f" hit=SUCCESS targets={hit_indices}")
    else:
        print(" hit=FAIL targets=[]")

    for idx, delta in enumerate(deltas):
        status = "HIT" if delta > 1e-6 else "MISS"
        print(
            f"  P{idx}: hp {player_hp_before[idx]:5.1f}->{player_hp_after[idx]:5.1f} "
            f"delta={delta:5.1f} {status}"
        )

    print(
        f" players_alive={info['players_alive']} kill_count={info.get('kill_count', 0)} "
        f"kill_reward={info.get('kill_reward', 0.0):4.2f} "
        f"skill_balance={info.get('skill_balance_reward', 0.0):5.2f} "
        f"skill_counts={list(info.get('skill_usage_counts', []))} "
        f"uniformity={info.get('skill_uniformity', 0.0):4.2f} "
        f"cycle={'Y' if info.get('skill_cycle_completed', False) else 'N'} "
        f"dot_ticks={list(info.get('dragon_dot_ticks', []))} | {format_players(env)}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", choices=["brass", "dragon"], default="brass")
    parser.add_argument(
        "--mode",
        choices=["manual", "random", "policy-greedy", "policy-sample", "sequence"],
        default="manual",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints_targeted")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--actions",
        default="",
        help="Comma-separated target:choice pairs, e.g. 0:0,1:3,2:4",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scripted_actions = []
    for item in args.actions.split(","):
        item = item.strip()
        if not item:
            continue
        target, choice = item.split(":", maxsplit=1)
        scripted_actions.append((int(target), int(choice)))

    env = BossPatternEnv(BossEnvConfig(seed=args.seed, boss_kind=args.boss))
    actor = None
    if args.mode.startswith("policy"):
        checkpoint = args.checkpoint or find_latest_checkpoint(args.boss, args.checkpoint_dir)
        if checkpoint is None:
            raise FileNotFoundError("no checkpoint found for policy mode")
        actor = load_actor(checkpoint, env)
        print(f"loaded checkpoint: {checkpoint}")

    print(
        f"boss={args.boss} mode={args.mode} map={env.cfg.map_size:.0f}x{env.cfg.map_size:.0f} "
        f"obs_dim={env.obs_dim} target_dim={env.target_dim} choice_dim={env.choice_dim} "
        f"steps={args.steps} episodes={args.episodes} delay={args.delay}"
    )
    print("separate output heads: target 0~2, choice 0~4")
    for choice, name in SKILL_NAMES[args.boss].items():
        print(f"  choice {choice}: {name}")

    for episode_idx in range(1, args.episodes + 1):
        obs, _ = env.reset()
        print(f"\n===== episode {episode_idx} start =====")
        print(f"boss_hp={env.boss_hp:.1f} boss_pos=({env.boss_pos[0]:.1f},{env.boss_pos[1]:.1f})")
        print(format_players(env))

        for step_idx in range(1, args.steps + 1):
            boss_hp_before = float(env.boss_hp)
            player_hp_before = env.player_hp.copy()
            action = select_action(args.mode, env, obs, actor, step_idx, scripted_actions)
            obs, reward, done, trunc, info = env.step(action)
            print_step_summary(
                env,
                args.boss,
                step_idx,
                action,
                reward,
                info,
                boss_hp_before,
                player_hp_before,
            )

            if done or trunc:
                if env.boss_hp <= 0.0:
                    result = "BOSS LOSE"
                elif info["players_alive"] <= 0:
                    result = "BOSS WIN"
                else:
                    result = "TIMEOUT"
                print(f"\n===== episode {episode_idx} end: {result} =====")
                break

            if args.delay > 0:
                time.sleep(args.delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped by user")
