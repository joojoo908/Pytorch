# --- Test.py ---

import ENV
import Model
import torch
import os

new = 1  # 1: 새 학습, 0: 이어 학습
ckpt_path = "sac_checkpoint.pth"

env = ENV.Vector2DEnv(
    geodesic_shaping=True,
    geodesic_progress_mode="delta",
    geodesic_coef=0.3,
    step_size=0.25,

    # ★ 근접 패널티 on (기본값 그대로여도 됨)
    proximity_penalty=True,
    proximity_threshold=0.20,
    proximity_coef=0.3,        # 너무 크면 목표 보상에 비해 학습이 소심해질 수 있음
    proximity_clip=0.2,

    # ★ 충돌 종료 off (기본값 False)
    collision_terminate=True,
    seed=28
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if new or not os.path.exists(ckpt_path):
    print("▶ 새로 학습 시작")
    bundle = Model.sac_train(env, episodes=100)

    Model.save_sac_checkpoint(
        ckpt_path,
        bundle["actor"], bundle["critic_1"], bundle["critic_2"],
        bundle["target_critic_1"], bundle["target_critic_2"],
        bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
        replay_buffer=bundle["replay_buffer"]
    )
    print(f"체크포인트 저장 완료: {ckpt_path}")

else:
    print(f"▶ 체크포인트 로드 및 이어 학습: {ckpt_path}")
    bundle = Model.load_sac_checkpoint(ckpt_path, state_dim, action_dim)

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
        episodes=1000
    )

    Model.save_sac_checkpoint(
        ckpt_path,
        bundle["actor"], bundle["critic_1"], bundle["critic_2"],
        bundle["target_critic_1"], bundle["target_critic_2"],
        bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
        replay_buffer=bundle["replay_buffer"]
    )
    print(f"추가 학습 후 체크포인트 저장 완료: {ckpt_path}")
