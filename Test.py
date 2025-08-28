# --- Test.py ---

import ENV
import Model
import torch
import os

new = 1  # 1: 새 학습, 0: 이어 학습
ckpt_path = "sac_checkpoint.pth"

env = ENV.Vector2DEnv(
    step_size=0.1,
    astar_grid=(256,256),
    astar_replan_steps=8,        # 게이팅
    replan_cte_threshold_frac=0.5,
    on_collision="slide",        # 정책이 회전 학습하게 유지
    # 보상 기본값은 이미 재밸런스됨(필요시 조절)
    # obs_with_extras=False 로 체크포인트와 입력차원 동일 유지
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
