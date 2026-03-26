# eval_ma_ppo.py
import numpy as np
import torch

from arena_env_ma import MultiAgentArenaEnv
from ppo_model import GaussianPolicyPPO, ValueNetwork, to_tensor

def load_ckpt(path, actor_p, critic_p, actor_m, critic_m, device):
    ck = torch.load(path, map_location=device)
    actor_p.load_state_dict(ck["actor_p"])
    critic_p.load_state_dict(ck["critic_p"])
    actor_m.load_state_dict(ck["actor_m"])
    critic_m.load_state_dict(ck["critic_m"])

@torch.no_grad()
def run_episode(env, actor_p, actor_m, device, render=False):
    obs, _ = env.reset()
    total = {"P": 0.0, "M": 0.0}

    for t in range(env.max_steps):
        action_dict = {}
        for aid, o in obs.items():
            alive = (o[5] > 0.5)
            if not alive:
                action_dict[aid] = np.zeros((3,), dtype=np.float32)
                continue

            x = to_tensor(o, device).unsqueeze(0)
            if aid.startswith("P"):
                a, logp, ent = actor_p.sample(x)
            else:
                a, logp, ent = actor_m.sample(x)
            action_dict[aid] = a.squeeze(0).cpu().numpy().astype(np.float32)

        obs, rew, term, trunc, info = env.step(action_dict)

        # 팀 리워드 합산(평가용)
        for aid, r in rew.items():
            if aid.startswith("P"):
                total["P"] += float(r)
            else:
                total["M"] += float(r)

        done_any = False
        for aid in obs.keys():
            if term[aid] or trunc[aid]:
                done_any = True
                break
        if done_any:
            break

    # 승패 판정
    alive_p = int(np.sum(env.p_hp > 0))
    alive_m = int(np.sum(env.m_hp > 0))
    if alive_m == 0 and alive_p > 0:
        outcome = "PLAYER_WIN"
    elif alive_p == 0 and alive_m > 0:
        outcome = "MONSTER_WIN"
    else:
        outcome = "DRAW"
    return total, outcome, alive_p, alive_m

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obstacles = [
        (0.0, 0.0, 1.2),
        (-2.5, 2.0, 0.8),
        ( 2.5,-2.0, 0.8),
    ]
    env = MultiAgentArenaEnv(seed=123, obstacles=obstacles, max_steps=512)

    obs_dim = env.obs_dim
    act_dim = env.act_dim
    actor_p = GaussianPolicyPPO(obs_dim, act_dim).to(device)
    critic_p = ValueNetwork(obs_dim).to(device)
    actor_m = GaussianPolicyPPO(obs_dim, act_dim).to(device)
    critic_m = ValueNetwork(obs_dim).to(device)

    load_ckpt("checkpoints/ma_ppo_latest.pth", actor_p, critic_p, actor_m, critic_m, device)

    wins = {"PLAYER_WIN": 0, "MONSTER_WIN": 0, "DRAW": 0}
    for ep in range(50):
        total, outcome, ap, am = run_episode(env, actor_p, actor_m, device)
        wins[outcome] += 1
        print(f"[EP {ep}] {outcome}  aliveP={ap} aliveM={am}  R_P={total['P']:.2f} R_M={total['M']:.2f}")

    print("Summary:", wins)

if __name__ == "__main__":
    main()
