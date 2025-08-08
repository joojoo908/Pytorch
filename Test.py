# --- Test.py ---

import ENV
import Model
import torch
import os

new = 1  # 1: 새 학습, 0: 이어 학습
ckpt_path = "sac_checkpoint.pth"

env = ENV.Vector2DEnv(map_range=12.8, step_size=0.1)
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
        replay_buffer=bundle["replay_buffer"]          # <<< 버퍼 저장
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
        replay_buffer=bundle["replay_buffer"],        # <<< 버퍼 전달
        episodes=100
    )

    Model.save_sac_checkpoint(
        ckpt_path,
        bundle["actor"], bundle["critic_1"], bundle["critic_2"],
        bundle["target_critic_1"], bundle["target_critic_2"],
        bundle["actor_opt"], bundle["critic_1_opt"], bundle["critic_2_opt"],
        replay_buffer=bundle["replay_buffer"]          # <<< 버퍼 저장
    )
    print(f"추가 학습 후 체크포인트 저장 완료: {ckpt_path}")
