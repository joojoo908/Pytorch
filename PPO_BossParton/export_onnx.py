import argparse
import os

import onnx
import torch

from ppo_model import TargetedCategoricalPolicyPPO


def find_latest_checkpoint(boss_kind: str, checkpoint_dir: str) -> str:
    prefix = f"{boss_kind}_targeted_ppo_"
    candidates = sorted(
        name
        for name in os.listdir(checkpoint_dir)
        if name.startswith(prefix) and name.endswith(".pth")
    )
    if not candidates:
        raise FileNotFoundError(f"no {boss_kind} checkpoint in {checkpoint_dir}")
    return os.path.join(checkpoint_dir, candidates[-1])


def export_checkpoint(checkpoint_path: str, output_path: str, boss_kind: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["actor"]
    obs_dim = int(state_dict["fc1.weight"].shape[1])
    target_dim = int(state_dict["target_logits.weight"].shape[0])
    choice_dim = int(state_dict["choice_logits.weight"].shape[0])

    actor = TargetedCategoricalPolicyPPO(obs_dim, target_dim, choice_dim)
    actor.load_state_dict(state_dict)
    actor.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_observation = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy_observation,
        output_path,
        input_names=["observation"],
        output_names=["target_logits", "choice_logits"],
        dynamic_axes={
            "observation": {0: "batch"},
            "target_logits": {0: "batch"},
            "choice_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    model = onnx.load(output_path)
    metadata = {
        "boss_kind": boss_kind,
        "source_checkpoint": os.path.basename(checkpoint_path),
        "observation_dim": str(obs_dim),
        "target_dim": str(target_dim),
        "choice_dim": str(choice_dim),
        "action_format": "target=argmax(target_logits), choice=argmax(choice_logits)",
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(
        f"exported {boss_kind}: {checkpoint_path} -> {output_path} "
        f"input=[batch,{obs_dim}] outputs=[batch,{target_dim}],[batch,{choice_dim}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", choices=["brass", "dragon", "both"], default="both")
    parser.add_argument("--checkpoint-dir", default="checkpoints_targeted")
    parser.add_argument("--output-dir", default="onnx_models")
    args = parser.parse_args()

    boss_kinds = ("brass", "dragon") if args.boss == "both" else (args.boss,)
    for boss_kind in boss_kinds:
        checkpoint_path = find_latest_checkpoint(boss_kind, args.checkpoint_dir)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(args.output_dir, f"{checkpoint_name}.onnx")
        export_checkpoint(checkpoint_path, output_path, boss_kind)


if __name__ == "__main__":
    main()
