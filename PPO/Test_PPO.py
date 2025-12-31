# Test_PPO.py
# -----------------------------------------
# [PPO 전용 학습 실행 스크립트]
# - 기존 Test.py는 SAC 학습용이므로 그대로 두고,
#   PPO는 이 파일로 돌릴 수 있게 분리했습니다.
# -----------------------------------------

import os
import argparse
import numpy as np
import torch

def build_env():
    """
    ENV.py에 Vector2DEnv가 정의되어 있으면 그것을 사용합니다.
    없다면 CartPole-v1로 폴백.
    """
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser(description="PPO training runner")
    # 기본 학습 파라미터
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)

    # PPO 하이퍼파라미터
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)

    ap.add_argument("--best-actor-path", type=str, default="ppo_actor_best.pth")
    args = ap.parse_args()

    set_global_seed(args.seed)

    import Model_PPO as ModelPPO

    env = build_env()

    # ENV에 dynamic_horizon 옵션이 있으면 참고 출력만
    if hasattr(env, "dynamic_horizon"):
        print(f"[INFO] ENV dynamic_horizon={env.dynamic_horizon} (PPO에서는 필수는 아님)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MODE] PPO | episodes={args.episodes} | max_steps={args.max_steps}")
    print(f"[CKPT] best_actor={args.best_actor_path}")

    ModelPPO.ppo_train(
        env=env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        epochs=args.epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
        best_actor_path=args.best_actor_path,
    )

if __name__ == "__main__":
    main()
