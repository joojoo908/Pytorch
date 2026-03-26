# train_ma_ppo.py
import os
import numpy as np
import torch

from arena_env_ma import MultiAgentArenaEnv
from ppo_model import GaussianPolicyPPO, ValueNetwork, to_tensor, compute_gae

class TeamBuffer:
    """
    팀(플레이어/몬스터)별로 transition을 모으는 버퍼.
    alive agent만 수집 (dead는 수집 안함)
    """
    def __init__(self):
        self.s = []
        self.a = []
        self.logp = []
        self.r = []
        self.done = []
        self.v = []

    def add(self, s, a, logp, r, done, v):
        self.s.append(s)
        self.a.append(a)
        self.logp.append(logp)
        self.r.append(r)
        self.done.append(done)
        self.v.append(v)

    def size(self):
        return len(self.r)

    def clear(self):
        self.__init__()

def rollout(env, actor_p, critic_p, actor_m, critic_m, device, T=256):
    obs, _ = env.reset()

    bufP = TeamBuffer()
    bufM = TeamBuffer()

    # rollout 중 마지막 bootstrap용 value 추정치(팀별)
    last_vP = 0.0
    last_vM = 0.0

    for t in range(T):
        action_dict = {}

        # 현재 step에서 alive인 agent만 행동/수집
        for aid, o in obs.items():
            alive = (o[5] > 0.5)  # 관측에서 alive flag
            if not alive:
                # dead agent action은 넣어도 무시되지만, 여기서는 그냥 0으로 둠
                action_dict[aid] = np.zeros((3,), dtype=np.float32)
                continue

            x = to_tensor(o, device).unsqueeze(0)

            if aid.startswith("P"):
                with torch.no_grad():
                    v = float(critic_p(x).item())
                    a, logp, ent = actor_p.sample(x)
                a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
                action_dict[aid] = a_np
                # reward/done은 step 후에 넣어야 하므로 임시 저장을 위해 step 후 다시 add할 때 사용
                # 여기서는 state/action/logp/value를 저장할 수 있게 임시 리스트로 들고 간다.
            else:
                with torch.no_grad():
                    v = float(critic_m(x).item())
                    a, logp, ent = actor_m.sample(x)
                a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
                action_dict[aid] = a_np

        next_obs, rew, term, trunc, info = env.step(action_dict)

        done_any = False
        # 환경에서 team 종료를 모든 agent에게 term=True로 주므로, 하나만 봐도 됨
        for aid in next_obs.keys():
            if term[aid] or trunc[aid]:
                done_any = True
                break

        # 이제 transition을 버퍼에 넣는다.
        # 주의: 위에서 action을 샘플링할 때 저장해두지 않았으니
        # 같은 계산을 다시 하면 안됨(정책이 바뀔 수 있음). 따라서 아래에서는
        # "이미 샘플링한 action/logp/value"가 필요하다.
        #
        # 해결: 위 루프에서 임시 dict로 저장해두자.
        # (아래에서 다시 샘플링하면 PPO가 깨진다.)
        #
        # -> 그래서 코드를 수정:
        #    action_dict 외에 sampled dict를 저장한다.

        # [수정] sampled 정보를 저장하도록 위 루프를 다시 구현
        # (설명: PPO는 old_logp가 반드시 '실제로 뽑은 action'의 logp여야 한다)

        obs = next_obs
        if done_any:
            break

    # 위에서 중간 수정이 필요하므로, 실제 동작하는 rollout을 아래에 "정상 버전"으로 다시 제공한다.
    raise RuntimeError("이 파일의 rollout은 아래 'rollout_fixed'를 사용하세요.")

def rollout_fixed(env, actor_p, critic_p, actor_m, critic_m, device, T=256):
    obs, _ = env.reset()

    bufP = TeamBuffer()
    bufM = TeamBuffer()

    for t in range(T):
        action_dict = {}
        sampled = {}  # aid -> (state, action, logp, value)

        for aid, o in obs.items():
            alive = (o[5] > 0.5)
            if not alive:
                action_dict[aid] = np.zeros((3,), dtype=np.float32)
                continue

            x = to_tensor(o, device).unsqueeze(0)
            if aid.startswith("P"):
                with torch.no_grad():
                    v = float(critic_p(x).item())
                    a, logp, ent = actor_p.sample(x)
                a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
                action_dict[aid] = a_np
                sampled[aid] = (o.astype(np.float32), a_np, float(logp.item()), v)
            else:
                with torch.no_grad():
                    v = float(critic_m(x).item())
                    a, logp, ent = actor_m.sample(x)
                a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
                action_dict[aid] = a_np
                sampled[aid] = (o.astype(np.float32), a_np, float(logp.item()), v)

        next_obs, rew, term, trunc, info = env.step(action_dict)

        done_any = False
        for aid in next_obs.keys():
            if term[aid] or trunc[aid]:
                done_any = True
                break

        # 버퍼에 추가(alive였던 agent만)
        for aid, (s, a, logp, v) in sampled.items():
            r = float(rew[aid])
            d = 1.0 if done_any else 0.0
            if aid.startswith("P"):
                bufP.add(s, a, logp, r, d, v)
            else:
                bufM.add(s, a, logp, r, d, v)

        obs = next_obs
        if done_any:
            break

    # bootstrap value: rollout 끝 상태에서 팀별로 "아무 alive agent 하나"를 대표로 잡아 value를 넣는 방식은 부정확함.
    # 여기서는 간단하게 bootstrap=0으로 두고(done_any가 자주 발생하므로) 안정성을 우선한다.
    # 더 정확히 하려면 팀별로 마지막 상태의 각 transition에 대해 V(s_{t+1})를 저장해야 한다.
    return bufP, bufM

def ppo_update(buf: TeamBuffer, actor, critic, actor_opt, critic_opt, device,
               gamma=0.99, lam=0.95,
               clip_eps=0.2,
               epochs=10,
               batch_size=2048,
               entropy_coef=0.0,
               value_coef=0.5,
               max_grad_norm=0.5):

    n = buf.size()
    if n < 128:
        return

    s = np.asarray(buf.s, dtype=np.float32)
    a = np.asarray(buf.a, dtype=np.float32)
    old_logp = np.asarray(buf.logp, dtype=np.float32)
    r = np.asarray(buf.r, dtype=np.float32)
    d = np.asarray(buf.done, dtype=np.float32)
    v = np.asarray(buf.v, dtype=np.float32)

    # bootstrap=0
    v_ext = np.concatenate([v, np.array([0.0], dtype=np.float32)], axis=0)
    adv, ret = compute_gae(r, v_ext, d, gamma=gamma, lam=lam)

    # normalize adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    s_t = to_tensor(s, device)
    a_t = to_tensor(a, device)
    old_logp_t = to_tensor(old_logp, device)
    adv_t = to_tensor(adv, device)
    ret_t = to_tensor(ret, device)

    idx_all = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx_all)

        for start in range(0, n, batch_size):
            idx = idx_all[start:start + batch_size]
            bs = s_t[idx]
            ba = a_t[idx]
            bold = old_logp_t[idx]
            badv = adv_t[idx]
            bret = ret_t[idx]

            new_logp, entropy = actor.evaluate_actions(bs, ba)
            ratio = torch.exp(new_logp - bold)

            surr1 = ratio * badv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * badv
            actor_loss = -torch.min(surr1, surr2).mean()

            v_pred = critic(bs)
            value_loss = (bret - v_pred).pow(2).mean()

            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            actor_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

            actor_opt.step()
            critic_opt.step()

def save_ckpt(path, actor_p, critic_p, actor_m, critic_m):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "actor_p": actor_p.state_dict(),
        "critic_p": critic_p.state_dict(),
        "actor_m": actor_m.state_dict(),
        "critic_m": critic_m.state_dict(),
    }, path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 장애물 예시(없애려면 obstacles=[]로)
    obstacles = [
        (0.0, 0.0, 1.2),
        (-2.5, 2.0, 0.8),
        ( 2.5,-2.0, 0.8),
    ]

    env = MultiAgentArenaEnv(
        seed=1,
        obstacles=obstacles,
        max_steps=512,
        monsters_min=10,
        monsters_max=20,
        k_nearest=3,
    )

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    actor_p = GaussianPolicyPPO(obs_dim, act_dim).to(device)
    critic_p = ValueNetwork(obs_dim).to(device)
    actor_m = GaussianPolicyPPO(obs_dim, act_dim).to(device)
    critic_m = ValueNetwork(obs_dim).to(device)

    opt_ap = torch.optim.Adam(actor_p.parameters(), lr=3e-4)
    opt_cp = torch.optim.Adam(critic_p.parameters(), lr=3e-4)
    opt_am = torch.optim.Adam(actor_m.parameters(), lr=3e-4)
    opt_cm = torch.optim.Adam(critic_m.parameters(), lr=3e-4)

    iters = 50000
    T = 256

    # 교대 업데이트 주기
    switch_every = 10  # 10 iter마다 학습 대상 팀 전환

    # PPO 파라미터
    ppo_kwargs = dict(
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=10,
        batch_size=2048,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
    )

    for it in range(1, iters + 1):
        bufP, bufM = rollout_fixed(env, actor_p, critic_p, actor_m, critic_m, device, T=T)

        # 어떤 팀을 업데이트할지 결정
        if ((it // switch_every) % 2) == 0:
            ppo_update(bufP, actor_p, critic_p, opt_ap, opt_cp, device, **ppo_kwargs)
        else:
            ppo_update(bufM, actor_m, critic_m, opt_am, opt_cm, device, **ppo_kwargs)

        if it % 50 == 0:
            print(f"[Iter {it}] P_samples={bufP.size()}  M_samples={bufM.size()}  monsters={env.monsters_n}")

        if it % 500 == 0:
            save_ckpt("checkpoints/ma_ppo_latest.pth", actor_p, critic_p, actor_m, critic_m)
            print("Saved: checkpoints/ma_ppo_latest.pth")

    save_ckpt("checkpoints/ma_ppo_final.pth", actor_p, critic_p, actor_m, critic_m)
    print("Saved: checkpoints/ma_ppo_final.pth")

if __name__ == "__main__":
    main()
