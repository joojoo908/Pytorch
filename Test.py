# Test.py — BC 사전학습 + SAC 파인튜닝
# ▶ PyCharm에서 인자 넣기 귀찮을 때: 코드 상단 CONFIG만 바꾸면 됩니다.
#   USE_CODE_CONFIG=True 면 아래 CONFIG 값이 곧바로 적용되고, argparse는 무시됩니다.

import os
import argparse
from types import SimpleNamespace
import numpy as np
import torch

import ENV
import Model
from Model import GaussianPolicy, device

# =========================
# ▼ 여기만 바꿔서 쓰세요 (필요시)
# =========================
USE_CODE_CONFIG = True  # True면 아래 CONFIG 사용, False면 CLI 인자 사용
CONFIG = SimpleNamespace(
    # 공통
    seed=1,
    random_maps=False,  # True면 랜덤 맵에서 수집/학습, False면 고정 맵

    # BC 설정
    bc_episodes=800,
    bc_epochs=10,
    bc_noise=0.05,

    # RL 설정
    rl_episodes=10000,
    target_entropy=-2.0,  # 연속 2D 액션이면 -2.0 권장

    # α 제어
    auto_alpha=True,          # True면 자동튜닝 (fixed_alpha가 None일 때 권장)
    fixed_alpha=None,         # 고정하고 싶으면 예: 0.18 (지정 시 auto_alpha는 무시됨)
    alpha_min=0.03,
    alpha_max=1.50,
    freeze_alpha_success=None,  # 최근100 성공률이 이 값 이상이면 α 그 시점에 고정

    # B안: 성공률 → target_entropy 보정
    success_target=0.80,   # 목표 성공률(0~1)
    te_lr=0.08,            # 보정 속도(너무 크면 진동)
    te_min=None,           # 기본: -2*action_dim
    te_max=None,           # 기본: -0.05*action_dim

    # Early Stop 옵션
    early_stop_success=0.98,      # 최근100 성공률 80% 이상이면
    early_stop_patience=3,        # 3회 연속 만족 시 종료
    early_stop_min_episodes=1000,  # 300 에피소드부터 체크 시작

    # 체크포인트/데모
    resume=False,
    ckpt_path="sac_checkpoint.pth",
    actor_path="sac_actor.pth",
    save_demo="",           # 예: "demos/demo1.npz"
    load_demo="",           # 예: "demos/demo1.npz"
)

# =========================
# BC(Behavior Cloning) 유틸
# =========================
def expert_action_geodesic(env, speed_gain=0.8):
    """지오데식 맵을 '내려가는' 전문가 정책. 반환: [-1,1]^2 액션"""
    geo = getattr(env, "_geo_map", None)
    if geo is None:
        vec = env.goal_pos - env.agent_pos
    else:
        r, c = env._pos_to_geo_rc(env.agent_pos)
        rows, cols = geo.shape
        best = (float(geo[r, c]) if np.isfinite(geo[r, c]) else np.inf, None)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
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

    ang = float(np.arctan2(vec[1], vec[0]))
    dist = float(np.linalg.norm(vec))
    speed_norm = float(np.clip(speed_gain * dist / max(getattr(env, "step_size", 1.0), 1e-6), 0.0, 1.0))
    a0 = float(np.clip(ang / np.pi, -1.0, 1.0))
    a1 = float(2.0 * speed_norm - 1.0)
    return np.array([a0, a1], dtype=np.float32)


def collect_bc_dataset(env, episodes=300, noise_std=0.05, max_steps=None):
    """전문가 정책으로 (obs, action) 쌍 수집."""
    Xs, Ys = [], []
    steps_limit = max_steps or getattr(env, "max_steps", 300)
    for _ in range(int(episodes)):
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
    """actor의 mean에 tanh를 씌워 [-1,1]^2로 만든 값을 데모 액션과 MSE로 맞춤."""
    import torch.nn.functional as F
    from torch import optim

    if dataset is None:
        X, Y = collect_bc_dataset(env, episodes=300, noise_std=0.05)
    else:
        X, Y = dataset

    actor.train()
    opt = optim.Adam(actor.parameters(), lr=lr)
    N = X.shape[0]
    for ep in range(int(epochs)):
        perm = np.random.permutation(N)
        tot = 0.0
        for i in range(0, N, int(batch_size)):
            idx = perm[i:i+int(batch_size)]
            xb = torch.from_numpy(X[idx]).float().to(device)
            yb = torch.from_numpy(Y[idx]).float().to(device)

            out = actor.forward(xb)
            mean = out[0] if isinstance(out, (tuple, list)) else out
            pred = torch.tanh(mean)

            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(idx)
        print(f"[BC] epoch {ep+1}/{epochs}  loss={tot/N:.4f}")
    return actor


# =========================
# 환경 생성
# =========================
def make_env(seed=1, fixed_maze=True):
    return ENV.Vector2DEnv(
        seed=seed,
        fixed_maze=fixed_maze,
        fixed_agent_goal=False,
        geodesic_shaping=True,
        geodesic_grid=(512, 512),
        proximity_penalty=False,
        proximity_threshold=0.0,
        proximity_coef=0.0,
        stall_penalty_use=True,
        stall_patience=5,
        stall_penalty_per_step=1.0,
        collision_terminate=True,
    )


# =========================
# 인자 병합 로직
# =========================
def parse_or_config():
    """USE_CODE_CONFIG=True면 CONFIG를 Namespace로 반환. 아니면 argparse 사용."""
    if USE_CODE_CONFIG:
        # fixed_alpha가 지정되어 있으면 자동튜닝을 끔
        if CONFIG.fixed_alpha is not None:
            CONFIG.auto_alpha = False
        return CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--random-maps", action="store_true", default=CONFIG.random_maps,
                        help="랜덤 맵 분포에서 학습/데이터수집")

    # BC 설정
    parser.add_argument("--bc-episodes", type=int, default=CONFIG.bc_episodes)
    parser.add_argument("--bc-epochs", type=int, default=CONFIG.bc_epochs)
    parser.add_argument("--bc-noise", type=float, default=CONFIG.bc_noise)

    # RL 설정
    parser.add_argument("--rl-episodes", type=int, default=CONFIG.rl_episodes)
    parser.add_argument("--target-entropy", type=float, default=CONFIG.target_entropy,
                        help="SAC 엔트로피 타겟")

    # α 제어 옵션(Model.py와 일치)
    parser.add_argument("--auto-alpha", action="store_true", default=CONFIG.auto_alpha,
                        help="α 자동튜닝 사용")
    parser.add_argument("--fixed-alpha", type=float, default=CONFIG.fixed_alpha,
                        help="자동튜닝을 끄고 이 값으로 α 고정")
    parser.add_argument("--alpha-min", type=float, default=CONFIG.alpha_min)
    parser.add_argument("--alpha-max", type=float, default=CONFIG.alpha_max)
    parser.add_argument("--freeze-alpha-success", type=float, default=CONFIG.freeze_alpha_success,
                        help="최근100 성공률이 이 값 이상이면 α 고정")

    # B안 파라미터
    parser.add_argument("--success-target", type=float, default=CONFIG.success_target,
                        help="최근 성공률 목표(0~1), 목표보다 낮으면 탐색↑")
    parser.add_argument("--te-lr", type=float, default=CONFIG.te_lr,
                        help="target_entropy 보정 속도")
    parser.add_argument("--te-min", type=float, default=CONFIG.te_min,
                        help="target_entropy 하한(기본: -2*action_dim)")
    parser.add_argument("--te-max", type=float, default=CONFIG.te_max,
                        help="target_entropy 상한(기본: -0.05*action_dim)")

    # Early Stop
    parser.add_argument("--early-stop-success", type=float, default=CONFIG.early_stop_success)
    parser.add_argument("--early-stop-patience", type=int,   default=CONFIG.early_stop_patience)
    parser.add_argument("--early-stop-min-episodes", type=int, default=CONFIG.early_stop_min_episodes)

    # 체크포인트
    parser.add_argument("--resume", action="store_true", default=CONFIG.resume,
                        help="체크포인트 이어 학습 (BC 건너뜀)")
    parser.add_argument("--ckpt-path", type=str, default=CONFIG.ckpt_path)
    parser.add_argument("--actor-path", type=str, default=CONFIG.actor_path)

    # 데모 저장/로드
    parser.add_argument("--save-demo", type=str, default=CONFIG.save_demo)
    parser.add_argument("--load-demo", type=str, default=CONFIG.load_demo)

    args = parser.parse_args()
    # 고정 α 지정 시 자동튜닝 비활성화
    if args.fixed_alpha is not None:
        args.auto_alpha = False
    return args


# =========================
# 메인 실행
# =========================
def main():
    args = parse_or_config()

    fixed_maze = not args.random_maps

    # -----------------
    # 이어 학습 모드
    # -----------------
    if args.resume and os.path.exists(args.ckpt_path):
        print(f"[Resume] 체크포인트에서 이어 학습: {args.ckpt_path}")
        env = make_env(seed=args.seed, fixed_maze=fixed_maze)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        bundle = Model.load_sac_checkpoint(args.ckpt_path, state_dim, action_dim)

        # 전달 옵션 구성(저장된 모드 우선, CLI/CONFIG로 덮어쓰기 허용)
        auto_alpha = (bundle.get("alpha_mode", "auto") == "auto") if args.fixed_alpha is None else False
        fixed_alpha = args.fixed_alpha if args.fixed_alpha is not None else bundle.get("fixed_alpha", None)

        bundle = Model.sac_train(
            env,
            actor=bundle["actor"],
            critic_1=bundle["critic_1"],
            critic_2=bundle["critic_2"],
            target_critic_1=bundle["target_critic_1"],
            target_critic_2=bundle["target_critic_2"],
            actor_opt=bundle["actor_opt"],
            critic_1_opt=bundle["critic_1_opt"],
            critic_2_opt=bundle["critic_2_opt"],
            replay_buffer=bundle["replay_buffer"],
            episodes=args.rl_episodes,
            log_alpha=bundle["log_alpha"],
            alpha_opt=bundle["alpha_opt"],
            target_entropy=(args.target_entropy if args.target_entropy is not None else bundle.get("target_entropy")),
            auto_alpha=auto_alpha,
            fixed_alpha=fixed_alpha,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            freeze_alpha_success=args.freeze_alpha_success,
            # Early Stop
            early_stop_success=args.early_stop_success,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_episodes=args.early_stop_min_episodes,
            # B안
            success_target=args.success_target,
            te_lr=args.te_lr,
            te_min=args.te_min,
            te_max=args.te_max,
        )

        Model.save_sac_checkpoint(
            args.ckpt_path,
            bundle["actor"], bundle["critic_1"], bundle["critic_2"],
            bundle["target_critic_1"], bundle["target_critic_2"],
            bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
            replay_buffer=bundle["replay_buffer"],
            log_alpha=bundle["log_alpha"],
            alpha_opt_state=bundle["alpha_opt"].state_dict(),
            target_entropy=bundle.get("target_entropy"),
            alpha_mode=bundle.get("alpha_mode"),
            fixed_alpha=bundle.get("fixed_alpha"),
        )
        torch.save(bundle["actor"].state_dict(), args.actor_path)
        print(f"[Resume] 저장 완료: ckpt={args.ckpt_path}, actor={args.actor_path}")
        return

    # -----------------
    # 신규: BC 사전학습 → SAC 파인튜닝
    # -----------------
    env = make_env(seed=args.seed, fixed_maze=fixed_maze)

    # 1) 데모 로드/수집
    if args.load_demo and os.path.exists(args.load_demo):
        print(f"[BC] 데모 로드: {args.load_demo}")
        data = np.load(args.load_demo)
        X_demo, A_demo = data["X"], data["A"]
    else:
        print(f"[BC] 데모 수집: episodes={args.bc_episodes}, random_maps={not fixed_maze}")
        X_demo, A_demo = collect_bc_dataset(env, episodes=args.bc_episodes, noise_std=args.bc_noise)
        if args.save_demo:
            os.makedirs(os.path.dirname(args.save_demo) or ".", exist_ok=True)
            np.savez_compressed(args.save_demo, X=X_demo, A=A_demo)
            print(f"[BC] 데모 저장 완료: {args.save_demo}")

    # 2) Actor 생성 → BC 사전학습
    actor = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    actor = bc_pretrain_actor(env, actor, dataset=(X_demo, A_demo), epochs=args.bc_epochs, batch_size=256)

    # 3) RL 파인튜닝 — Model.py 새 인터페이스 사용
    auto_alpha = args.auto_alpha if args.fixed_alpha is None else False
    bundle = Model.sac_train(
        env,
        actor=actor,
        episodes=args.rl_episodes,
        target_entropy=args.target_entropy,
        auto_alpha=auto_alpha,
        fixed_alpha=args.fixed_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        freeze_alpha_success=args.freeze_alpha_success,
        # Early Stop
        early_stop_success=args.early_stop_success,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_episodes=args.early_stop_min_episodes,
        # B안
        success_target=args.success_target,
        te_lr=args.te_lr,
        te_min=args.te_min,
        te_max=args.te_max,
    )

    # 4) 저장(전체 ckpt + actor만) — α 상태 포함
    Model.save_sac_checkpoint(
        args.ckpt_path,
        bundle["actor"], bundle["critic_1"], bundle["critic_2"],
        bundle["target_critic_1"], bundle["target_critic_2"],
        bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
        replay_buffer=bundle["replay_buffer"],
        log_alpha=bundle["log_alpha"],
        alpha_opt_state=bundle["alpha_opt"].state_dict(),
        target_entropy=bundle["target_entropy"],
        alpha_mode=bundle.get("alpha_mode"),
        fixed_alpha=bundle.get("fixed_alpha"),
    )
    torch.save(bundle["actor"].state_dict(), args.actor_path)
    print(f"[Done] 저장 완료: ckpt={args.ckpt_path}, actor={args.actor_path}")


if __name__ == "__main__":
    main()
