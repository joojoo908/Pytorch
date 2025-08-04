import sys
print("▶ PyCharm이 쓰는 Python:", sys.executable)

import numpy as np
import random
import pygame
import time

# -----------------------------
# 환경 설정
GRID_SIZE = 10
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_TO_DELTA = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# -----------------------------
# 헬퍼 함수
def move(pos, action):
    dx, dy = ACTION_TO_DELTA[action]
    x, y = pos[0] + dx, pos[1] + dy
    x = max(0, min(GRID_SIZE - 1, x))
    y = max(0, min(GRID_SIZE - 1, y))
    return (x, y)

def get_state(agent_pos, player_pos):
    return (agent_pos[0], agent_pos[1], player_pos[0], player_pos[1])

def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# -----------------------------
# Q-learning 파라미터 및 함수
Q = {}
alpha = 0.5         # 학습률
gamma = 0.9         # 할인율
epsilon = 0.2       # 탐험률

def select_action(state):
    if random.random() < epsilon or state not in Q:
        return random.choice(ACTIONS)
    return max(Q[state], key=Q[state].get)

def update_q(state, action, reward, next_state):
    if state not in Q:
        Q[state] = {a: 0.0 for a in ACTIONS}
    if next_state not in Q:
        Q[next_state] = {a: 0.0 for a in ACTIONS}

    max_future = max(Q[next_state].values())
    Q[state][action] += alpha * (reward + gamma * max_future - Q[state][action])

# -----------------------------
# 학습 루프
EPISODES = 10000
success_count = 0

for episode in range(EPISODES):
    agent_pos = (0, 0)
    player_pos = (GRID_SIZE - 1, GRID_SIZE - 1)

    for step in range(100):
        state = get_state(agent_pos, player_pos)
        action = select_action(state)
        next_agent_pos = move(agent_pos, action)

        # 플레이어 무작위 이동
        player_action = random.choice(ACTIONS)
        player_pos = move(player_pos, player_action)

        next_state = get_state(next_agent_pos, player_pos)
        dist = manhattan_dist(next_agent_pos, player_pos)
        reward = 1.0 if next_agent_pos == player_pos else -0.1 * dist / GRID_SIZE

        update_q(state, action, reward, next_state)
        agent_pos = next_agent_pos

        if agent_pos == player_pos:
            success_count += 1
            break

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Success rate (last 100) = {success_count}/100")
        success_count = 0

# -----------------------------
# 시각화 (pygame)
pygame.init()
cell_size = 50
screen = pygame.display.set_mode((GRID_SIZE * cell_size, GRID_SIZE * cell_size))
pygame.display.set_caption("강화학습 기반 추적 AI")
clock = pygame.time.Clock()

def draw(agent_pos, player_pos):
    screen.fill((255, 255, 255))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    # 플레이어 (빨간색)
    pygame.draw.rect(screen, (255, 0, 0), (player_pos[0]*cell_size, player_pos[1]*cell_size, cell_size, cell_size))
    # 에이전트 (파란색)
    pygame.draw.rect(screen, (0, 0, 255), (agent_pos[0]*cell_size, agent_pos[1]*cell_size, cell_size, cell_size))

    pygame.display.flip()

def run_simulation():
    agent_pos = (0, 0)
    player_pos = (GRID_SIZE - 1, GRID_SIZE - 1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 에이전트 행동 (탐험 없음)
        state = get_state(agent_pos, player_pos)
        if state in Q:
            action = max(Q[state], key=Q[state].get)
        else:
            action = random.choice(ACTIONS)

        agent_pos = move(agent_pos, action)

        # 플레이어 무작위 이동
        player_pos = move(player_pos, random.choice(ACTIONS))

        draw(agent_pos, player_pos)
        clock.tick(5)

        if agent_pos == player_pos:
            print("🎯 잡았다!")
            time.sleep(1)
            agent_pos = (0, 0)
            player_pos = (GRID_SIZE - 1, GRID_SIZE - 1)

run_simulation()
