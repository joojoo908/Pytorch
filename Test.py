# Test.py — BC(모방학습) 사전학습 + SAC 파인튜닝 + 이어학습(α 자동튜닝 포함)
# 사용 예시:
#   1) BC 사전학습 후 RL 파인튜닝(고정 맵):
#      python Test.py --bc-episodes 400 --bc-epochs 10 --rl-episodes 10000
#   2) 랜덤 맵 분포에서 BC/RL 모두 학습:
#      python Test.py --random-maps --bc-episodes 600 --rl-episodes 12000
#   3) 기존 체크포인트 이어 학습(RL만):
#      python Test.py --resume --rl-episodes 10000
#
# 메모:
# - 이 파일은 Model.py에 BC 유틸이 없어도 동작하도록 BC 함수를 자체 포함합니다.
# - ENV 옵션(충돌/지오데식/보상 등)은 학습·평가·데이터수집에서 일관되게 설정하는 것이 중요합니다.

import os
import argparse
import numpy as np
import torch

import ENV
import Model
from Model import GaussianPolicy, device

# =========================
# BC(Behavior Cloning) 유틸
# =========================

def expert_action_geodesic(env, speed_gain=0.8):
    """
    지오데식 맵을 '내려가는' 전문가 정책.
    반환: [-1,1]^2 액션 (a0=각도/π, a1=속도스케일)
    """
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

    ang = float(np.arctan2(vec[1], vec[0]))                     # [-π, π]
    dist = float(np.linalg.norm(vec))
    speed_norm = float(np.clip(speed_gain * dist / max(env.step_size, 1e-6), 0.0, 1.0))
    a0 = float(np.clip(ang / np.pi, -1.0, 1.0))                 # 각도 [-1,1]
    a1 = float(2.0 * speed_norm - 1.0)                          # 속도 [-1,1]
    return np.array([a0, a1], dtype=np.float32)

def collect_bc_dataset(env, episodes=300, noise_std=0.05, max_steps=None):
    """
    전문가 정책으로 (obs, action) 쌍 수집.
    noise_std: 레이블에 소노이즈(일반화 향상)
    """
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
    """
    actor의 mean에 tanh를 씌워 [-1,1]^2로 만든 값을 데모 액션과 MSE로 맞춤.
    """
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

            out = actor.forward(xb)          # (mean, log_std) 또는 튜플
            mean = out[0] if isinstance(out, (tuple, list)) else out
            pred = torch.tanh(mean)          # [-1,1]^2

            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(idx)
        print(f"[BC] epoch {ep+1}/{epochs}  loss={tot/N:.4f}")
    return actor

# =========================
# 환경 생성
# =========================

def make_env(seed=1, fixed_maze=True):
    """
    학습/평가/데이터수집 모두 동일한 규칙으로 생성.
    필요한 경우 옵션을 여기서 통일하세요.
    """
    return ENV.Vector2DEnv(
        seed=seed,
        fixed_maze=fixed_maze,          # True: 고정 맵, False: 매 에피소드 새 맵
        fixed_agent_goal=False,
        geodesic_shaping=True,
        geodesic_grid=(512, 512),
        proximity_penalty=False,         # 벽 근접 억제
        proximity_threshold=0.15,
        proximity_coef=0.5,
        stall_penalty_use=True,         # 정체 억제
        stall_patience=5,
        stall_penalty_per_step=1.0,
        collision_terminate=True,
    )

# =========================
# 메인 실행
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-maps", action="store_true", help="랜덤 맵 분포에서 학습/데이터수집")
    parser.add_argument("--bc-episodes", type=int, default=5000, help="BC 데이터 수집 에피소드 수")
    parser.add_argument("--bc-epochs", type=int, default=30, help="BC 사전학습 에폭 수")
    parser.add_argument("--bc-noise", type=float, default=0.05, help="BC 레이블 노이즈 표준편차")
    parser.add_argument("--rl-episodes", type=int, default=30000, help="SAC 학습 에피소드 수")
    parser.add_argument("--resume", action="store_true", help="체크포인트 이어 학습 (BC 건너뜀)")
    parser.add_argument("--ckpt-path", type=str, default="sac_checkpoint.pth")
    parser.add_argument("--actor-path", type=str, default="sac_actor.pth")
    parser.add_argument("--save-demo", type=str, default="", help="수집한 BC 데모를 .npz로 저장(선택)")
    parser.add_argument("--load-demo", type=str, default="", help=".npz 데모 파일에서 로드(선택)")
    args = parser.parse_args()

    fixed_maze = not args.random_maps

    # -----------------
    # 이어 학습 모드
    # -----------------
    if 0 and os.path.exists(args.ckpt_path):
        print(f"[Resume] 체크포인트에서 이어 학습: {args.ckpt_path}")
        env = make_env(seed=args.seed, fixed_maze=fixed_maze)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        bundle = Model.load_sac_checkpoint(args.ckpt_path, state_dim, action_dim)

        # 동일 분포의 환경으로 RL 진행 (+ α 상태 주입)
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
            target_entropy=bundle.get("target_entropy", -float(action_dim)),
        )

        # 저장 (α 상태 포함)
        Model.save_sac_checkpoint(
            args.ckpt_path,
            bundle["actor"], bundle["critic_1"], bundle["critic_2"],
            bundle["target_critic_1"], bundle["target_critic_2"],
            bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
            replay_buffer=bundle["replay_buffer"],
            log_alpha=bundle["log_alpha"],
            alpha_opt_state=bundle["alpha_opt"].state_dict(),
            target_entropy=bundle.get("target_entropy", -float(action_dim)),
        )
        torch.save(bundle["actor"].state_dict(), args.actor_path)
        print(f"[Resume] 저장 완료: ckpt={args.ckpt_path}, actor={args.actor_path}")
        return

    # -----------------
    # 신규: BC 사전학습 → SAC 파인튜닝
    # -----------------
    env = make_env(seed=args.seed, fixed_maze=fixed_maze)

    # 2) (선택) 데모 로드 또는 수집
    if args.load_demo and os.path.exists(args.load_demo):
        print(f"[BC] 데모 로드: {args.load_demo}")
        data = np.load(args.load_demo)
        X_demo, A_demo = data["X"], data["A"]
    else:
        print(f"[BC] 데모 수집: episodes={args.bc_episodes}, random_maps={not fixed_maze}")
        X_demo, A_demo = collect_bc_dataset(env, episodes=args.bc_episodes, noise_std=args.bc_noise)
        if args.save_demo:
            np.savez_compressed(args.save_demo, X=X_demo, A=A_demo)
            print(f"[BC] 데모 저장 완료: {args.save_demo}")

    # 3) Actor 생성 → BC 사전학습
    actor = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    actor = bc_pretrain_actor(env, actor, dataset=(X_demo, A_demo), epochs=args.bc_epochs, batch_size=256)

    # 4) RL 파인튜닝 (리플레이 버퍼는 새로 시작)
    bundle = Model.sac_train(env, actor=actor, episodes=args.rl_episodes)

    # 5) 저장(전체 ckpt + actor만) — α 상태 포함
    Model.save_sac_checkpoint(
        args.ckpt_path,
        bundle["actor"], bundle["critic_1"], bundle["critic_2"],
        bundle["target_critic_1"], bundle["target_critic_2"],
        bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
        replay_buffer=bundle["replay_buffer"],
        log_alpha=bundle["log_alpha"],
        alpha_opt_state=bundle["alpha_opt"].state_dict(),
        target_entropy=bundle["target_entropy"],
    )
    torch.save(bundle["actor"].state_dict(), args.actor_path)
    print(f"[Done] 저장 완료: ckpt={args.ckpt_path}, actor={args.actor_path}")


if __name__ == "__main__":
    main()