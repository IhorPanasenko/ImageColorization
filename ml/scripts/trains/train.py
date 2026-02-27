"""
Unified training entry point — all 4 colorization stages.

Usage examples:
    python scripts/trains/train.py --model baseline --epochs 20 --batch_size 16
    python scripts/trains/train.py --model unet     --epochs 20 --batch_size 16
    python scripts/trains/train.py --model gan      --epochs 20 --batch_size 8 --lambda_l1 100
    python scripts/trains/train.py --model fusion   --epochs 20 --batch_size 8 --lambda_l1 100

Dispatches to the appropriate per-stage training function.
All arguments accepted by individual training scripts are supported here.
"""
import sys
import os
import argparse

# Add scripts/trains/ to sys.path so sibling module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add project root so src.* imports work from this location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def get_args():
    parser = argparse.ArgumentParser(
        description="Unified Colorization Trainer — Stages 1–4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      type=str,   default="unet",
                        choices=["baseline", "unet", "gan", "fusion"],
                        help="Architecture to train")
    parser.add_argument("--epochs",     type=int,   default=20,                      help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,                       help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-4,                    help="Initial learning rate")
    parser.add_argument("--lambda_l1",  type=float, default=100.0,                   help="L1 loss weight (Stages 3 & 4)")
    parser.add_argument("--data_path",  type=str,   default="./data/coco/val2017",   help="Path to training images")
    parser.add_argument("--save_dir",   type=str,   default="./outputs/checkpoints", help="Checkpoint output directory")
    parser.add_argument("--log_dir",    type=str,   default="./outputs/runs",        help="TensorBoard log directory")
    # Resume flags
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Checkpoint to resume from (Stages 1 & 2)")
    parser.add_argument("--resume_g",   type=str,   default=None,
                        help="Generator checkpoint to resume from (Stages 3 & 4)")
    parser.add_argument("--resume_d",   type=str,   default=None,
                        help="Discriminator checkpoint to resume from (Stages 3 & 4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"=== Unified Trainer | Model: {args.model.upper()} | Epochs: {args.epochs} ===")

    if args.model == "baseline":
        from train_baseline import train_baseline
        train_baseline(args)

    elif args.model == "unet":
        from train_unet import train_unet
        train_unet(args)

    elif args.model == "gan":
        from train_gan import train_gan
        train_gan(args)

    elif args.model == "fusion":
        from train_fusion import train_fusion
        train_fusion(args)