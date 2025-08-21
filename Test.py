# --- Test.py ---

import ENV
import Model
import torch
import os

new = 1  # 1: 새 학습, 0: 이어 학습
ckpt_path = "sac_checkpoint.pth"

env = ENV.Vector2DEnv(
    maze_cells=(15, 15),
    step_size=0.1,
    on_collision="deflect",
    R_SUCCESS=500.0,
    # A* 셀 길이 보상(새로 추가된 옵션)
    astar_shaping_scale=2.0,
    astar_shaping_clip=5.0,
    astar_grid=(256,256),           # 전역 A* 격자 해상도
    astar_replan_steps=1            # A* 재계획 주기(스텝)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if new or not os.path.exists(ckpt_path):
    print("▶ 새로 학습 시작")
    bundle = Model.sac_train(env, episodes=10)

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
