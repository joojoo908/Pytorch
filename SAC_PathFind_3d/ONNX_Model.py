from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Model


class ActorDeterministic(nn.Module):
    def __init__(self, base_actor: nn.Module):
        super().__init__()
        self.base = base_actor

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.base(state)
        return torch.tanh(mean)


def infer_dims_from_gaussian_policy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        first_weight = state_dict["net.0.weight"]
        last_weight = state_dict["net.6.weight"]
    except KeyError as exc:
        raise RuntimeError(
            "GaussianPolicy state_dict format is not recognized. "
            "Expected keys such as 'net.0.weight' and 'net.6.weight'."
        ) from exc

    obs_dim = int(first_weight.shape[1])
    act_dim = int(last_weight.shape[0] // 2)
    if obs_dim <= 0 or act_dim <= 0:
        raise RuntimeError(f"Invalid inferred dims: obs_dim={obs_dim}, act_dim={act_dim}")
    return obs_dim, act_dim


def load_multi_role_actor(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected checkpoint dict, got {type(obj).__name__}")
    if obj.get("format") != "multi_role_actor":
        raise RuntimeError(f"Expected format='multi_role_actor', got {obj.get('format')!r}")
    actors = obj.get("actors")
    if not isinstance(actors, dict):
        raise RuntimeError("Checkpoint is missing an 'actors' dict.")
    return actors


def export_role_actor(
    role_name: str,
    state_dict: Dict[str, torch.Tensor],
    out_path: Path,
    opset: int,
    obs_dim_override: int | None = None,
    act_dim_override: int | None = None,
) -> Tuple[int, int]:
    inferred_obs_dim, inferred_act_dim = infer_dims_from_gaussian_policy_state_dict(state_dict)
    obs_dim = int(obs_dim_override if obs_dim_override is not None else inferred_obs_dim)
    act_dim = int(act_dim_override if act_dim_override is not None else inferred_act_dim)

    if obs_dim != inferred_obs_dim or act_dim != inferred_act_dim:
        raise RuntimeError(
            f"{role_name}: explicit dims do not match checkpoint dims. "
            f"explicit=({obs_dim}, {act_dim}), checkpoint=({inferred_obs_dim}, {inferred_act_dim})"
        )

    actor = Model.GaussianPolicy(obs_dim, act_dim).cpu().eval()
    actor.load_state_dict(state_dict)
    wrapped = ActorDeterministic(actor).cpu().eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["state"],
        output_names=["action"],
        dynamic_axes={"state": {0: "batch"}, "action": {0: "batch"}},
    )
    return obs_dim, act_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multi-role SAC actors to role-specific ONNX files.")
    parser.add_argument("--actor-path", type=str, default="sac_actor_best.pth", help="Path to multi_role_actor checkpoint.")
    parser.add_argument("--out-dir", type=str, default="onnx_roles", help="Directory to write role ONNX files.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--obs-dim", type=int, default=None, help="Optional explicit observation dim. Must match checkpoint.")
    parser.add_argument("--act-dim", type=int, default=None, help="Optional explicit action dim. Must match checkpoint.")
    args = parser.parse_args()

    actor_path = Path(args.actor_path)
    if not actor_path.is_absolute():
        actor_path = SCRIPT_DIR / actor_path
    if not actor_path.exists():
        raise SystemExit(f"Actor checkpoint not found: {actor_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir

    actors = load_multi_role_actor(actor_path)
    exported = []
    missing_roles = []

    for role_id in Model.ROLE_IDS:
        role_name = Model.role_name(role_id)
        state_dict = actors.get(role_name)
        if state_dict is None:
            missing_roles.append(role_name)
            continue
        out_path = out_dir / f"{role_name}.onnx"
        obs_dim, act_dim = export_role_actor(
            role_name=role_name,
            state_dict=state_dict,
            out_path=out_path,
            opset=args.opset,
            obs_dim_override=args.obs_dim,
            act_dim_override=args.act_dim,
        )
        exported.append((role_name, out_path, obs_dim, act_dim))

    if missing_roles:
        raise SystemExit(f"Checkpoint is missing role actors: {', '.join(missing_roles)}")

    print("ONNX export complete")
    print(f"source: {actor_path}")
    print(f"out_dir: {out_dir}")
    for role_name, out_path, obs_dim, act_dim in exported:
        print(f"- {role_name}: {out_path.name}  input=float32[batch,{obs_dim}] output=float32[batch,{act_dim}]")
    print("none(-1) is not exported; handle it as fallback/no-model in C++.")


if __name__ == "__main__":
    main()
