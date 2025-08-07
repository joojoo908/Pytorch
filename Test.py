import ENV
import Model

import torch


env = ENV.Vector2DEnv()
actor = Model.sac_train(env, episodes=1000)  # 약 1000 에피소드 학습

torch.save(actor.state_dict(), "sac_actor.pth")
print("모델저장완료")
