# viewer_watch_only.py
import numpy as np
import torch
import pygame

from arena_env_ma import MultiAgentArenaEnv
from ppo_model import GaussianPolicyPPO, ValueNetwork, to_tensor


def load_ckpt(path, actor_p, critic_p, actor_m, critic_m, device):
    ck = torch.load(path, map_location=device)
    actor_p.load_state_dict(ck["actor_p"])
    critic_p.load_state_dict(ck["critic_p"])
    actor_m.load_state_dict(ck["actor_m"])
    critic_m.load_state_dict(ck["critic_m"])


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ------------------------------
# [수정] UI 레이아웃/좌표계
# ------------------------------
ARENA_PAD = 20  # 왼쪽 아레나 테두리 패딩(기존 border 20px에 맞춤)


def world_to_screen(p, map_range, arena_w, arena_h):
    """
    [수정됨]
    world: [-map_range, map_range] -> 화면(왼쪽 아레나 영역)으로 변환
    - 전체 창 크기(W,H)가 아니라, 아레나 영역(ARENA_W, ARENA_H) 기준으로만 매핑
    - 패딩(ARENA_PAD) 고려
    """
    usable_w = arena_w - 2 * ARENA_PAD
    usable_h = arena_h - 2 * ARENA_PAD

    x = (p[0] / map_range * 0.5 + 0.5) * usable_w + ARENA_PAD
    y = (-(p[1] / map_range) * 0.5 + 0.5) * usable_h + ARENA_PAD
    return int(x), int(y)


def radius_to_screen(r, map_range, arena_w):
    """
    [수정됨]
    반지름도 아레나 영역 폭 기준으로 스케일
    """
    usable_w = arena_w - 2 * ARENA_PAD
    return max(1, int(r / (2 * map_range) * usable_w))


# ------------------------------
# 정책 액션 (관전용)
# ------------------------------
DETERMINISTIC = True
# True  = 결정적 관전( mean 기반 ) : 움직임이 깔끔하지만 a2가 0이하면 공격 안 할 수 있음
# False = 확률적 관전( sample 기반 ) : 학습 시 행동처럼 보여서 더 “살아있는” 경우가 많음


@torch.no_grad()
def act_policy(actor: GaussianPolicyPPO, obs: np.ndarray, device):
    x = to_tensor(obs.astype(np.float32), device).unsqueeze(0)

    if DETERMINISTIC:
        # [중요] act_deterministic()가 모델에 없을 수 있으므로 forward+tanh로 직접 처리
        mu = actor.forward(x)
        a = torch.tanh(mu)
    else:
        # 학습 때처럼 sample() 사용
        a, _, _ = actor.sample(x)

    return a.squeeze(0).cpu().numpy().astype(np.float32)


# ------------------------------
# [추가] 디버그 패널 렌더링
# ------------------------------
def draw_debug_panel(screen, font, small_font, env, action_dict, ckpt_path, panel_x, panel_w, H):
    """
    [추가된 함수]
    오른쪽 디버그 패널:
      - env 내부 상태를 실시간 출력
      - action_dict에 들어있는 현재 프레임 액션도 같이 출력(디버깅에 유용)
    """
    # 패널 배경/분리선
    pygame.draw.rect(screen, (24, 24, 28), pygame.Rect(panel_x, 0, panel_w, H))
    pygame.draw.line(screen, (70, 70, 75), (panel_x, 0), (panel_x, H), 2)

    x = panel_x + 12
    y = 12
    line_h = 18

    def blit_line(text, color=(230, 230, 230), use_small=True):
        nonlocal y
        f = small_font if use_small else font
        screen.blit(f.render(text, True, color), (x, y))
        y += line_h

    # 헤더
    blit_line("=== DEBUG PANEL ===", (255, 220, 120), use_small=False)
    blit_line(f"ckpt: {ckpt_path}", (200, 200, 200))
    blit_line(f"step={env.steps} / {env.max_steps}   paused={getattr(env, 'paused', False)}", (200, 200, 200))

    alive_p = int(np.sum(env.p_hp > 0))
    alive_m = int(np.sum(env.m_hp > 0))
    blit_line(f"aliveP={alive_p}/{env.n_players}   aliveM={alive_m}/{env.monsters_n}", (200, 200, 200))
    blit_line(f"DETERMINISTIC={DETERMINISTIC}", (200, 200, 200))
    blit_line("")

    # 플레이어 상태
    blit_line("[Players]", (180, 220, 255))
    for i in range(env.n_players):
        hp = float(env.p_hp[i])
        alive = 1 if hp > 0 else 0
        pos = env.p_pos[i]
        cd = int(env.p_cd[i]) if hasattr(env, "p_cd") else -1
        a = action_dict.get(f"P{i}", np.zeros((3,), dtype=np.float32))
        blit_line(
            f"P{i} alive={alive} hp={hp:6.1f} cd={cd:2d} "
            f"pos=({pos[0]:+.2f},{pos[1]:+.2f})  a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})"
        )

    blit_line("")

    # 몬스터 상태 (너무 많으니 일부만)
    blit_line("[Monsters] (lowest HP top 12)", (255, 180, 180))
    alive_idx = np.where(env.m_hp > 0)[0]
    if alive_idx.size == 0:
        blit_line("none", (160, 160, 160))
        return

    N = min(12, alive_idx.size)
    sorted_idx = alive_idx[np.argsort(env.m_hp[alive_idx])][:N]

    for mi in sorted_idx:
        hp = float(env.m_hp[mi])
        pos = env.m_pos[mi]
        cd = int(env.m_cd[mi]) if hasattr(env, "m_cd") else -1
        a = action_dict.get(f"M{mi}", np.zeros((3,), dtype=np.float32))
        blit_line(
            f"M{mi:02d} hp={hp:6.1f} cd={cd:2d} pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
            f"a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 장애물(원하면 [])
    obstacles = [
        (0.0, 0.0, 1.2),
        (-2.5, 2.0, 0.8),
        (2.5, -2.0, 0.8),
    ]

    env = MultiAgentArenaEnv(
        seed=123,
        obstacles=obstacles,
        max_steps=1024,
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

    # 체크포인트 경로
    ckpt_path = "checkpoints/ma_ppo_latest.pth"
    load_ckpt(ckpt_path, actor_p, critic_p, actor_m, critic_m, device)
    actor_p.eval(); critic_p.eval()
    actor_m.eval(); critic_m.eval()

    # ---------------- UI LAYOUT ----------------
    ARENA_W, ARENA_H = 900, 900
    PANEL_W = 420
    W, H = ARENA_W + PANEL_W, ARENA_H  # [수정] 오른쪽 패널 공간

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Watch Only Viewer + Debug Panel (MA PPO Arena)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 14)

    obs, _ = env.reset()

    # ------------------------------
    # (추가) 원거리 공격 트레이서(선) 표시용 버퍼
    #   - env.last_shots 에 기록된 ranged 이벤트를 받아서 몇 프레임 동안 선으로 그린다.
    # ------------------------------
    tracers = []  # [{'p0':np.array([x,y]), 'p1':np.array([x,y]), 'ttl':int, 'hit':bool}, ...]
    TRACER_TTL = 5

    paused = False
    env.paused = paused  # 패널에서 출력용

    running = True

    while running:
        clock.tick(60)

        # 이번 프레임 액션(패널 표시용)
        action_dict = {}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    env.paused = paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    paused = False
                    env.paused = paused

        if not paused:
            # players AI
            for i in range(env.n_players):
                aid = f"P{i}"
                o = obs[aid]
                if o[5] > 0.5:
                    action_dict[aid] = act_policy(actor_p, o, device)
                else:
                    action_dict[aid] = np.zeros((3,), dtype=np.float32)

            # monsters AI
            for i in range(env.monsters_n):
                aid = f"M{i}"
                o = obs[aid]
                if o[5] > 0.5:
                    action_dict[aid] = act_policy(actor_m, o, device)
                else:
                    action_dict[aid] = np.zeros((3,), dtype=np.float32)

            obs, rew, term, trunc, info = env.step(action_dict)

            # ------------------------------
            # (추가) 이번 step에서 발생한 원거리 샷 이벤트를 tracer로 등록
            #   env.last_shots: [{'kind':'ranged', 'start':..., 'end':..., 'hit':...}, ...]
            # ------------------------------
            for s in getattr(env, 'last_shots', []):
                if s.get('kind') != 'ranged':
                    continue
                tracers.append({
                    'p0': np.array(s['start'], dtype=np.float32),
                    'p1': np.array(s['end'], dtype=np.float32),
                    'ttl': TRACER_TTL,
                    'hit': bool(s.get('hit', True)),
                })


            # 종료 시 자동 일시정지
            done_any = False
            for aid in obs.keys():
                if term[aid] or trunc[aid]:
                    done_any = True
                    break
            if done_any:
                paused = True
                env.paused = paused

        # ---------------- render (왼쪽 아레나) ----------------
        screen.fill((18, 18, 22))

        # 아레나 border (왼쪽 영역만)
        pygame.draw.rect(
            screen,
            (70, 70, 75),
            pygame.Rect(ARENA_PAD, ARENA_PAD, ARENA_W - 2 * ARENA_PAD, ARENA_H - 2 * ARENA_PAD),
            2
        )

        # obstacles
        for (ox, oy, orad) in env.obstacles:
            sp = world_to_screen(np.array([ox, oy], dtype=np.float32), env.map_range, ARENA_W, ARENA_H)
            rr = radius_to_screen(orad, env.map_range, ARENA_W)
            pygame.draw.circle(screen, (90, 90, 90), sp, rr)


        # ------------------------------
        # (추가) 원거리 샷 트레이서 렌더링
        #   - hit=True : 노란색
        #   - hit=False: 회색(장애물에 막힘)
        # ------------------------------
        if tracers:
            alive_tracers = []
            for t in tracers:
                p0 = world_to_screen(t['p0'], env.map_range, ARENA_W, ARENA_H)
                p1 = world_to_screen(t['p1'], env.map_range, ARENA_W, ARENA_H)
                col = (255, 220, 80) if t.get('hit', True) else (140, 140, 140)
                pygame.draw.line(screen, col, p0, p1, 2)
                pygame.draw.circle(screen, col, p1, 3)

                t['ttl'] -= 1
                if t['ttl'] > 0:
                    alive_tracers.append(t)
            tracers = alive_tracers

        # players
        for i in range(env.n_players):
            hp = float(env.p_hp[i])
            pos = env.p_pos[i]
            sp = world_to_screen(pos, env.map_range, ARENA_W, ARENA_H)
            rr = radius_to_screen(env.radius_player, env.map_range, ARENA_W)

            alive = hp > 0
            col = (70, 190, 255) if alive else (40, 40, 40)
            pygame.draw.circle(screen, col, sp, rr)

            # HP bar
            bar_w, bar_h = 44, 6
            x0, y0 = sp[0] - bar_w // 2, sp[1] - rr - 14
            pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(x0, y0, bar_w, bar_h))
            frac = clamp01(hp / env.hp_player_max)
            pygame.draw.rect(screen, (200, 60, 60), pygame.Rect(x0, y0, int(bar_w * frac), bar_h))

            label = font.render(f"P{i}", True, (230, 230, 230))
            screen.blit(label, (sp[0] + rr + 4, sp[1] - rr - 2))

        # monsters
        for i in range(env.monsters_n):
            hp = float(env.m_hp[i])
            pos = env.m_pos[i]
            sp = world_to_screen(pos, env.map_range, ARENA_W, ARENA_H)
            rr = radius_to_screen(env.radius_monster, env.map_range, ARENA_W)

            alive = hp > 0
            col = (240, 90, 90) if alive else (35, 35, 35)
            pygame.draw.circle(screen, col, sp, rr)

            # HP bar small
            bar_w, bar_h = 28, 4
            x0, y0 = sp[0] - bar_w // 2, sp[1] - rr - 10
            pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(x0, y0, bar_w, bar_h))
            frac = clamp01(hp / env.hp_monster_max)
            pygame.draw.rect(screen, (200, 60, 60), pygame.Rect(x0, y0, int(bar_w * frac), bar_h))

        # HUD text (왼쪽 상단)
        alive_p = int(np.sum(env.p_hp > 0))
        alive_m = int(np.sum(env.m_hp > 0))
        status = f"step={env.steps}  aliveP={alive_p}/{env.n_players}  aliveM={alive_m}/{env.monsters_n}  paused={paused}"
        screen.blit(font.render(status, True, (230, 230, 230)), (25, 25))
        screen.blit(font.render("Keys: P pause/resume, R reset, ESC quit", True, (200, 200, 200)), (25, 50))

        if paused:
            if alive_m == 0 and alive_p > 0:
                msg = "PLAYER WIN (R to reset)"
            elif alive_p == 0 and alive_m > 0:
                msg = "MONSTER WIN (R to reset)"
            else:
                msg = "PAUSED / DRAW (R to reset)"
            screen.blit(font.render(msg, True, (255, 220, 120)), (25, 75))

        # ---------------- render (오른쪽 패널) ----------------
        panel_x = ARENA_W
        draw_debug_panel(
            screen=screen,
            font=font,
            small_font=small_font,
            env=env,
            action_dict=action_dict if not paused else {},
            ckpt_path=ckpt_path,
            panel_x=panel_x,
            panel_w=PANEL_W,
            H=H
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
