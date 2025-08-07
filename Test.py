import ENV
import Model  # sac_model.py를 Model 모듈로 불러온다고 가정

import torch


new = 0  # 1이면 새 학습, 0이면 이어 학습

# 환경 생성
env = ENV.Vector2DEnv(map_range=12.8, step_size=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 모델 생성
actor = Model.GaussianPolicy(state_dim, action_dim).to(device)
critic_1 = Model.QNetwork(state_dim, action_dim).to(device)
critic_2 = Model.QNetwork(state_dim, action_dim).to(device)
target_critic_1 = Model.QNetwork(state_dim, action_dim).to(device)
target_critic_2 = Model.QNetwork(state_dim, action_dim).to(device)

if new:
    # 초기 critic들도 target에 복사
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    # 학습 시작
    actor = Model.sac_train(
        env,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        target_critic_1=target_critic_1,
        target_critic_2=target_critic_2,
        episodes=1000
    )

    # actor 저장
    torch.save(actor.state_dict(), "sac_actor.pth")
    print("모델 저장 완료")

else:
    # 기존 모델 로드
    actor.load_state_dict(torch.load("sac_actor.pth", map_location=device))
    print("모델 불러오기 완료")

    # target critic도 초기화
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    # 추가 학습
    actor = Model.sac_train(
        env,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        target_critic_1=target_critic_1,
        target_critic_2=target_critic_2,
        episodes=500
    )

    # 저장
    torch.save(actor.state_dict(), "sac_actor.pth")
    print("추가 학습 완료 및 모델 저장 완료")
