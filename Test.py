import os
import argparse
import numpy as np
import torch

# ---- Build env from user's ENV module if present ----
def build_env():
    try:
        import ENV as ENV_mod
        if hasattr(ENV_mod, "make_env") and callable(ENV_mod.make_env):
            return ENV_mod.make_env()
        for name in ("Vector2DEnv", "Env"):
            if hasattr(ENV_mod, name):
                cls = getattr(ENV_mod, name)
                return cls() if callable(cls) else cls
    except Exception as e:
        print(f"[WARN] Using fallback env due to import error: {e}")
    try:
        import gymnasium as gym
    except Exception:
        import gym as gym
    return gym.make("CartPole-v1")

def set_global_seed(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser(description="PyCharm-friendly training runner (no CMD args needed).")
    # Core
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", default=False)     # may be auto-enabled
    ap.add_argument("--strict-resume", action="store_true", default=False)
    ap.add_argument("--ckpt", type=str, default="sac_best.pth")
    ap.add_argument("--no-auto-resume", action="store_true", default=False,
                    help="If set, never auto-resume even if ckpt exists.")

    # Success mixing
    ap.add_argument("--p-succ", type=float, default=0.30)
    ap.add_argument("--succ-buffer-cap", type=int, default=200_000)
    ap.add_argument("--succ-gate-min", type=int, default=2048)
    ap.add_argument("--succ-ramp-cov", type=float, default=0.25)
    ap.add_argument("--succ-min-dist", type=float, default=0.20)

    # Updates & exploration
    ap.add_argument("--updates-per-step", type=int, default=2)
    ap.add_argument("--alpha-floor", type=float, default=0.05)
    ap.add_argument("--alpha-ceiling", type=float, default=1.00)

    # Best saving
    ap.add_argument("--save-best-online", action="store_true", default=True)
    ap.add_argument("--best-delta", type=float, default=0.02)
    ap.add_argument("--best-min-episodes", type=int, default=100)
    ap.add_argument("--best-ckpt-path", type=str, default="sac_best.pth")
    ap.add_argument("--best-actor-path", type=str, default="sac_actor_best.pth")

    # Dynamic horizon (ENV-side) — default ON for PyCharm convenience
    ap.add_argument("--dyn-horizon", action="store_true", default=True)
    ap.add_argument("--dyn-kappa", type=float, default=1.6)
    ap.add_argument("--dyn-tmin", type=int, default=64)
    ap.add_argument("--dyn-tmax", type=int, default=2048)
    ap.add_argument("--dyn-geo", action="store_true", default=True)

    ap.add_argument("--alpha-freeze-recent", type=float, default=1.000)  # 0.40 = 40%
    ap.add_argument("--alpha-freeze-succbuf", type=int, default=150_000)
    ap.add_argument("--alpha-fixed", type=float, default=0.63)

    args = ap.parse_args()

    # Set seeds
    set_global_seed(args.seed)

    # Import training module
    import Model as Model

    # Build ENV
    env = build_env()
    # Apply dynamic horizon params if supported
    if hasattr(env, "dynamic_horizon"):
        env.dynamic_horizon = bool(args.dyn_horizon)
        env.dynamic_horizon_kappa = float(args.dyn_kappa)
        env.dynamic_horizon_Tmin = int(args.dyn_tmin)
        env.dynamic_horizon_Tmax = int(args.dyn_tmax)
        env.dynamic_horizon_use_geodesic = bool(args.dyn_geo)

    # Infer dims once (also initializes env internals)
    obs = Model.reset_env(env)

    # If user set a fixed max-steps, enforce it
    if args.max_steps is not None and hasattr(env, "max_steps"):
        env.max_steps = args.max_steps

    # ---- Auto-resume convenience for PyCharm ----
    auto_resume = False
    if (not args.no_auto_resume) and (not args.resume) and os.path.exists(args.ckpt):
        args.resume = True
        auto_resume = True

    # Load checkpoint if resuming
    bundle = None
    if args.resume:
        try:
            obs_dim = int(np.asarray(obs).shape[-1])
            act_dim = int(getattr(getattr(env, "action_space", None), "shape", [2])[0])
            bundle = Model.load_sac_checkpoint(args.ckpt, obs_dim, act_dim, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"[INFO] Loaded checkpoint from {args.ckpt}")
        except Exception as e:
            if args.strict_resume:
                raise
            print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")
            bundle = None

    # Pretty mode banner
    mode = "RESUME" if bundle else "FRESH"
    auto = " (auto)" if auto_resume else ""
    print(f"[MODE] {mode}{auto} | episodes={args.episodes} | batch={args.batch_size} | max_steps={args.max_steps} | dyn_horizon={getattr(env, 'dynamic_horizon', 'N/A')}")
    print(f"[CKPT] in/out: {args.ckpt} | best_ckpt={args.best_ckpt_path} | best_actor={args.best_actor_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training
    Model.sac_train(
        env=env,
        # actor=bundle["actor"],
        # critic_1=bundle["critic_1"],
        # critic_2=bundle["critic_2"],
        # target_critic_1=bundle["target_critic_1"],
        # target_critic_2=bundle["target_critic_2"],
        # actor_opt=bundle["actor_opt"],
        # critic_1_opt=bundle["critic_1_opt"],
        # critic_2_opt=bundle["critic_2_opt"],
        # replay_buffer=bundle["replay_buffer"],

        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        p_succ=args.p_succ,
        succ_buffer_capacity=args.succ_buffer_cap,
        succ_gate_min=args.succ_gate_min,
        succ_ramp_cov=args.succ_ramp_cov,
        succ_min_dist=args.succ_min_dist,
        updates_per_step=args.updates_per_step,
        alpha_floor=args.alpha_floor,
        alpha_ceiling=args.alpha_ceiling,
        alpha_freeze_recent=args.alpha_freeze_recent,
        alpha_freeze_succbuf=args.alpha_freeze_succbuf,
        alpha_fixed=args.alpha_fixed,
        save_best_online=args.save_best_online,
        best_delta=args.best_delta,
        best_min_episodes=args.best_min_episodes,
        best_ckpt_path=args.best_ckpt_path,
        best_actor_path=args.best_actor_path,
        bundle=bundle,
        device=device,
    )

if __name__ == "__main__":
    main()
