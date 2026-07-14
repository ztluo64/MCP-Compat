from pathlib import Path
import torch

from tri_view_compat.tools.eval_threshold_calibrated import build_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    ckpt_dir = Path("tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256")
    ckpt_path = ckpt_dir / "best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})

    model = build_model("prior_v2", args, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    total, trainable = count_params(model)

    print("===== Checkpoint =====")
    print("ckpt_path:", ckpt_path)
    print("epoch:", ckpt.get("epoch"))
    print()

    print("===== Checkpoint args =====")
    for k in sorted(args.keys()):
        print(f"{k}: {args[k]}")
    print()

    print("===== Parameter count =====")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")
    print()

    print("===== Model switches =====")
    print("use_full:", model.use_full)
    print("use_crop:", model.use_crop)
    print("use_masked:", model.use_masked)
    print("use_text:", model.use_text)
    print("use_geo:", model.use_geo)
    print("freeze_clip:", model.freeze_clip)
    print("num_prior_classes:", model.num_prior_classes)
    print("clip_dim:", model.clip_dim)
    print()

    print("===== Modules =====")
    print("prior_head:")
    print(model.prior_head)
    print()
    print("prior_proj:")
    print(model.prior_proj)
    print()
    print("text_prior_proj:")
    print(model.text_prior_proj)
    print()
    print("scalar_mlp:")
    print(model.scalar_mlp)
    print()
    print("classifier:")
    print(model.classifier)
    print()

    print("===== Trainable modules =====")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, tuple(p.shape), p.numel())


if __name__ == "__main__":
    main()
