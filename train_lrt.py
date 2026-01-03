import argparse
import os
import sys
from typing import Any, Dict, List


def _add_repo_python_to_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.join(repo_root, "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def _parse_paths(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(",")]
        out.extend([p for p in parts if p])
    return out


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    total_steps = int(args.epochs * args.steps_per_epoch)

    return {
        "seed": int(args.seed),
        "model": {
            "hidden_dim": int(args.hidden_dim),
            "num_heads": int(args.num_heads),
            "max_steps": int(args.max_reasoning_steps),
            "min_reasoning_steps": int(args.min_reasoning_steps),
            "dropout_rate": float(args.dropout_rate),
            "deterministic": True,
            "use_enhanced_encoder": bool(args.use_enhanced_encoder),
        },
        "training": {
            "learning_rate": float(args.learning_rate),
            "end_learning_rate": float(args.end_learning_rate),
            "warmup_steps": int(args.warmup_steps),
            "total_steps": int(total_steps),
            "steps_per_epoch": int(args.steps_per_epoch),
            "val_steps": int(args.val_steps),
            "batch_size": int(args.batch_size),
            "max_grad_norm": float(args.max_grad_norm),
            "weight_decay": float(args.weight_decay),
            "value_weight": float(args.value_weight),
            "policy_weight": float(args.policy_weight),
            "step_penalty": float(args.step_penalty),
            "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
            "save_every": int(args.save_every),
            "keep_checkpoints": int(args.keep_checkpoints),
        },
        "logging": {
            "use_wandb": bool(args.use_wandb),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True, help="PGN path(s). Repeatable, or comma-separated.")
    parser.add_argument("--val", action="append", default=[], help="Validation PGN path(s). Repeatable, or comma-separated.")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--keep-checkpoints", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--min-reasoning-steps", type=int, default=2)
    parser.add_argument("--max-reasoning-steps", type=int, default=32)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--use-enhanced-encoder", action="store_true",
                    help="Use enhanced board encoder with chess features")

    args = parser.parse_args()

    _add_repo_python_to_path()

    from liquid_chess.training.trainer import ChessDataset, LRTTrainer

    train_paths = _parse_paths(args.train)
    val_paths = _parse_paths(args.val)

    if not train_paths:
        raise SystemExit("No training PGN paths provided")

    cfg = build_config(args)

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    train_ds = ChessDataset(train_paths, batch_size=cfg["training"]["batch_size"], shuffle=True, use_enhanced_features=cfg["model"]["use_enhanced_encoder"])
    if val_paths:
        val_ds = ChessDataset(val_paths, batch_size=cfg["training"]["batch_size"], shuffle=False, use_enhanced_features=cfg["model"]["use_enhanced_encoder"])
    else:
        val_ds = train_ds

    trainer = LRTTrainer(cfg)
    trainer.train(train_ds, val_ds, num_epochs=int(args.epochs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

