# export_onnx.py  (auto-detect version)
# 그냥 실행:  python export_onnx.py
# 또는 명시 실행:
#   python export_onnx.py --ckpt sac_checkpoint.pth --out actor.onnx
#   python export_onnx.py --actor_only sac_actor.pth --out actor.onnx

import argparse
import os, sys, glob
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Model  # GaussianPolicy 정의를 사용

class ActorDeterministic(nn.Module):
    def __init__(self, base_actor: nn.Module):
        super().__init__()
        self.base = base_actor
    def forward(self, state):
        mean, std = self.base(state)
        return torch.tanh(mean)

def _infer_dims_from_state_dict(sd: dict):
    """
    Model.GaussianPolicy의 state_dict로부터 (state_dim, action_dim) 자동 추론
    - 첫 Linear: fc.0.weight  -> shape = [128, state_dim]
    - mean layer: mean.weight -> shape = [action_dim, 128]
    """
    try:
        w0 = sd["fc.0.weight"]
        w_mean = sd["mean.weight"]
        state_dim = w0.shape[1]
        action_dim = w_mean.shape[0]
        return int(state_dim), int(action_dim)
    except Exception as e:
        raise RuntimeError(f"입력/출력 차원 자동 추론 실패: {e}")

def _auto_find_checkpoint():
    """
    현재 폴더에서 유력 파일 자동 검색.
    우선순위:
      1) sac_actor.pth (actor 전용 저장)
      2) sac_checkpoint.pth (전체 체크포인트)
      3) *.pth 중 가장 최근 수정 파일
    """
    candidates = []
    if os.path.exists("sac_actor.pth"):
        candidates.append(("actor_only", "sac_actor.pth"))
    if os.path.exists("sac_checkpoint.pth"):
        candidates.append(("ckpt", "sac_checkpoint.pth"))

    if candidates:
        return candidates[0]  # 우선순위 1,2 중 첫 번째

    # 폴더 내 임의의 .pth 중 최신
    pths = sorted(glob.glob("*.pth"), key=lambda p: os.path.getmtime(p), reverse=True)
    if pths:
        # 어떤 형식인지 모르니 일단 전체 체크포인트일 가능성 먼저 시도
        return ("auto", pths[0])

    return (None, None)

def _load_actor_from_any(path: str, device: torch.device):
    """
    path가 actor-only인지 전체 ckpt인지 자동 판별해서 actor state_dict와 dims를 반환.
    """
    # PyTorch 2.6 이후 기본 weights_only=True 변화 대응: 안전하게 False로
    bundle = torch.load(path, map_location=device, weights_only=False)

    # 전체 체크포인트 형태(딕셔너리에 'actor' 키)인지 확인
    if isinstance(bundle, dict) and "actor" in bundle and isinstance(bundle["actor"], dict):
        actor_sd = bundle["actor"]
    else:
        # actor-only라고 가정
        actor_sd = bundle if isinstance(bundle, dict) else bundle.state_dict()

    state_dim, action_dim = _infer_dims_from_state_dict(actor_sd)

    # 실제 모듈 구성/로딩
    actor = Model.GaussianPolicy(state_dim, action_dim).to(device)
    actor.load_state_dict(actor_sd)
    actor.eval()
    return actor, state_dim, action_dim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--actor_only", type=str, default=None)
    ap.add_argument("--state_dim", type=int, default=None)   # 명시하면 자동추론 대신 우선
    ap.add_argument("--action_dim", type=int, default=None)  # 명시하면 자동추론 대신 우선
    ap.add_argument("--out", type=str, default="actor.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    device = torch.device("cpu")

    src_kind = None
    src_path = None
    if args.ckpt:
        src_kind = "ckpt"
        src_path = args.ckpt
    elif args.actor_only:
        src_kind = "actor_only"
        src_path = args.actor_only
    else:
        # 인자 없으면 자동탐지
        src_kind, src_path = _auto_find_checkpoint()
        if src_kind is None:
            raise SystemExit("모델 파일을 못 찾았습니다. (--ckpt 또는 --actor_only 인자 제공 필요)")

    if not os.path.exists(src_path):
        raise SystemExit(f"파일이 없습니다: {src_path}")

    # 로드 + 차원 자동 추론
    actor, inferred_s, inferred_a = _load_actor_from_any(src_path, device)

    # 사용자가 명시한 값이 있으면 우선
    state_dim = args.state_dim if args.state_dim is not None else inferred_s
    action_dim = args.action_dim if args.action_dim is not None else inferred_a

    # 래퍼 후 ONNX Export
    wrapped = ActorDeterministic(actor).to(device).eval()
    dummy = torch.zeros(1, state_dim, dtype=torch.float32, device=device)
    torch.onnx.export(
        wrapped, dummy, args.out,
        export_params=True, opset_version=args.opset, do_constant_folding=True,
        input_names=["state"], output_names=["action"],
        dynamic_axes={"state": {0: "batch"}, "action": {0: "batch"}},
    )

    print("✅ ONNX 내보내기 완료")
    print(f"   소스: {src_path} (kind={src_kind})")
    print(f"   dims: state_dim={state_dim}, action_dim={action_dim}")
    print(f"   ONNX: {args.out}, opset={args.opset}")
    print("   입력: float32 [batch, state_dim]  /  출력: float32 [batch, action_dim] in [-1,1]")

if __name__ == "__main__":
    main()
