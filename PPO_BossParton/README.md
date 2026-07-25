# PPO_BossParton

보스가 `대상 1명과 이동/스킬 행동`을 고르는 학습용 PPO 예제다.
`Brass`와 `Dragon`은 공용 모델이 아니라 각각 별도 모델로 학습한다.

## 파일

- `boss_pattern_env.py`
  - 보스 1명 vs 플레이어 3명 전투를 단순화한 학습 환경
  - `3000 x 3000` 맵, 보스와 플레이어 이동 포함
  - 관측 20차원: 전역 상태 5개 + 플레이어별 방향/거리/체력/생존 5개
  - 정책 출력: 대상 `Discrete(3)` + 행동 `Discrete(5)`의 독립 헤드
  - `choice 0`: 선택 대상에게 이동하고 스킬을 사용하지 않음
  - `choice 1~4`: 현재 위치에서 선택 대상을 향해 보스 스킬 1~4 사용
- `ppo_model.py`
  - Categorical PPO actor / critic
- `train_boss_ppo.py`
  - rollout 수집, GAE 계산, PPO 업데이트, 체크포인트 저장

## 실행

```bash
cd /home/joojoo/PPO_BossParton
python3 train_boss_ppo.py
```

개별 학습만 돌리고 싶으면:

```bash
python3 train_boss_ppo.py --boss brass
python3 train_boss_ppo.py --boss dragon
```

실시간으로 스킬 사용과 데미지 적용 여부를 확인하려면:

```bash
python3 live_pattern_test.py --boss brass --mode manual
python3 live_pattern_test.py --boss dragon --mode sequence --actions 0:0,0:1,1:4,2:4 --delay 1.0
python3 live_pattern_test.py --boss brass --mode policy-greedy
```

- `manual`: 매 턴 `대상번호 선택번호` 입력. 예: `2 4`
- `sequence`: 지정한 액션 시퀀스를 반복 실행
- `policy-greedy` / `policy-sample`: 체크포인트 정책으로 자동 실행
- 출력 내용:
  - 어떤 스킬을 사용했는지
  - 실제로 어느 플레이어가 맞았는지 (`HIT` / `MISS`)
  - 플레이어 HP 변화량
  - 보스가 준 피해 / 받은 피해

## 실제 서버 연결 시 교체할 부분

현재 환경은 토이 시뮬레이터다. 실제 연결 시에는 아래를 교체하면 된다.

1. `BossPatternEnv._build_obs()`
   - `EnemySystem` / `EnemyComponent` / 플레이어 상태에서 실제 관측 구성
2. `BossPatternEnv.step()`
   - 서버 전투 결과를 받아 reward 계산
3. `action = (target, choice)`
   - `target 0~2`와 `choice 0~4`를 별도 정책 헤드에서 선택
   - `choice 0`은 이동, `choice 1~4`는 보스 스킬에 매핑
