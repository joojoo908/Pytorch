# Test.py
# Minimal driver that wires your ENV + Model.sac_train with success-replay mixing options.
# Adjust the ENV import and constructor to your project as needed.

import argparse
import torch

import Model

# ---- device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_env():
    """
    Replace this with your own ENV construction.
    We try a few common entry points to reduce friction.
    """
    try:
        import ENV as ENV_mod
        # priority: make_env() -> Vector2DEnv() -> Env()
        if hasattr(ENV_mod, "make_env"):
            return ENV_mod.make_env()
        elif hasattr(ENV_mod, "Vector2DEnv"):
            return ENV_mod.Vector2DEnv()
        elif hasattr(ENV_mod, "Env"):
            return ENV_mod.Env()
        else:
            raise RuntimeError("ENV.py found but no known constructor (make_env/Vector2DEnv/Env)")
    except Exception as e:
        print("[WARN] Could not import your ENV.py properly:", e)
        print("       Falling back to a dummy random-walk gymnasium env for shape probing.")
        import gymnasium as gym
        return gym.make("CartPole-v1")  # You should NOT train SAC here; only to keep script runnable.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--ckpt", type=str, default="sac_best.pth")

    # success mixing
    ap.add_argument("--p-succ", type=float, default=0.30)
    ap.add_argument("--succ-buffer-cap", type=int, default=200_000)
    ap.add_argument("--succ-gate-min", type=int, default=2048)
    ap.add_argument("--succ-ramp-cov", type=float, default=0.25)
    ap.add_argument("--succ-min-dist", type=float, default=0.20)

    # updates & exploration
    ap.add_argument("--updates-per-step", type=int, default=2)
    ap.add_argument("--alpha-floor", type=float, default=0.05)
    ap.add_argument("--alpha-ceiling", type=float, default=1.00)

    # best saving
    ap.add_argument("--save-best-online", action="store_true", default=True)
    ap.add_argument("--best-delta", type=float, default=0.02)
    ap.add_argument("--best-min-episodes", type=int, default=30)
    ap.add_argument("--best-ckpt-path", type=str, default="sac_best.pth")
    ap.add_argument("--best-actor-path", type=str, default="sac_actor_best.pth")

    args = ap.parse_args()

    env = build_env()
    print("[DEBUG] Using env:", type(env).__name__)


    # Optionally resume from checkpoint
    bundle = None
    if args.resume:
        # Infer dimensions lazily for loader
        _probe = Model.reset_env(env)
        if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
            act_dim = int(env.action_space.shape[0])
        else:
            act_dim = 2
        if isinstance(_probe, (list, tuple, )):
            import numpy as np
            obs_dim = int(np.asarray(_probe).shape[-1])
        else:
            obs_dim = int(len(_probe))
        try:
            bundle = Model.load_sac_checkpoint(args.ckpt, obs_dim, act_dim, device=device)
            print(f"[INFO] Loaded checkpoint from {args.ckpt}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")
            bundle = None

    # Prepare buffers
    main_rb = bundle["replay_buffer"] if (bundle and "replay_buffer" in bundle) else Model.ReplayBuffer(capacity=1_000_000)
    succ_rb = Model.SuccessReplayBuffer(capacity=args.succ_buffer_cap)

    # Unpack networks/opts when resuming
    actor = bundle["actor"] if bundle else None
    critic_1 = bundle["critic_1"] if bundle else None
    critic_2 = bundle["critic_2"] if bundle else None
    target_critic_1 = bundle["target_critic_1"] if bundle else None
    target_critic_2 = bundle["target_critic_2"] if bundle else None
    actor_opt = bundle["actor_opt"] if bundle else None
    critic_1_opt = bundle["critic_1_opt"] if bundle else None
    critic_2_opt = bundle["critic_2_opt"] if bundle else None

    # Train
    out = Model.sac_train(
        env,
        actor=actor,
        critic_1=critic_1, critic_2=critic_2,
        target_critic_1=target_critic_1, target_critic_2=target_critic_2,
        actor_opt=actor_opt, critic_1_opt=critic_1_opt, critic_2_opt=critic_2_opt,
        replay_buffer=main_rb,
        succ_replay_buffer=succ_rb,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        # mixing
        p_succ=args.p_succ,
        succ_gate_min=args.succ_gate_min,
        succ_ramp_cov=args.succ_ramp_cov,
        succ_min_dist=args.succ_min_dist,
        # updates & exploration
        updates_per_step=args.updates_per_step,
        alpha_floor=args.alpha_floor,
        alpha_ceiling=args.alpha_ceiling,
        # best
        save_best_online=args.save_best_online,
        best_delta=args.best_delta,
        best_min_episodes=args.best_min_episodes,
        best_ckpt_path=args.best_ckpt_path,
        best_actor_path=args.best_actor_path,
        device=device,
    )

    # Final save (actor last already saved by Model.sac_train)
    print("[DONE] Training complete. Best actor:", args.best_actor_path)


if __name__ == "__main__":
    main()
