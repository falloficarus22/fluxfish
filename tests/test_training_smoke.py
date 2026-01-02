import time
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_python_to_path() -> None:
    import sys

    python_dir = _repo_root() / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def _write_tiny_pgn(path: Path) -> None:
    chess = pytest.importorskip("chess")
    chess_pgn = pytest.importorskip("chess.pgn")

    game = chess_pgn.Game()
    board = chess.Board()
    node = game

    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
        node = node.add_variation(move)

    game.headers["Result"] = "1/2-1/2"

    with path.open("w", encoding="utf-8") as f:
        print(game, file=f)


def test_training_step_updates_params(tmp_path: Path):
    _add_repo_python_to_path()

    import jax
    from jax import random

    from liquid_chess.training.trainer import ChessDataset, LRTTrainer

    pgn_path = tmp_path / "tiny.pgn"
    _write_tiny_pgn(pgn_path)

    cfg = {
        "seed": 0,
        "model": {
            "hidden_dim": 64,
            "num_heads": 4,
            "dropout_rate": 0.0,
            "min_reasoning_steps": 2,
            "max_steps": 4,
            "deterministic": True,
        },
        "training": {
            "learning_rate": 1e-3,
            "end_learning_rate": 1e-4,
            "warmup_steps": 0,
            "total_steps": 4,
            "steps_per_epoch": 1,
            "val_steps": 1,
            "batch_size": 2,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "value_weight": 1.0,
            "policy_weight": 1.0,
            "step_penalty": 0.0,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "save_every": 1000,
            "keep_checkpoints": 1,
        },
        "logging": {"use_wandb": False},
    }

    ds = ChessDataset([str(pgn_path)], batch_size=cfg["training"]["batch_size"], shuffle=True)
    trainer = LRTTrainer(cfg)
    trainer.state = trainer.create_train_state()

    batch = ds.get_batch()

    params_before = trainer.state.params

    rng = random.PRNGKey(0)

    t_compile0 = time.time()
    new_state, metrics = trainer.train_step(trainer.state, batch, rng)

    print(f"Loss: {float(metrics['loss'])}")
    print(f"Value loss: {float(metrics['value_loss'])}")  
    print(f"Policy loss: {float(metrics['policy_loss'])}")
    
    _ = float(metrics["loss"])
    compile_s = time.time() - t_compile0

    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: jax.numpy.isfinite(x).all(), metrics))

    def _tree_l1(a, b):
        leaves_a = jax.tree_util.tree_leaves(a)
        leaves_b = jax.tree_util.tree_leaves(b)
        return sum([float(jax.numpy.sum(jax.numpy.abs(x - y))) for x, y in zip(leaves_a, leaves_b)])

    delta = _tree_l1(params_before, new_state.params)
    assert delta > 0.0

    trainer.state = new_state

    t0 = time.time()
    for _ in range(2):
        trainer.state, _ = trainer.train_step(trainer.state, ds.get_batch(), rng)
    step_s = time.time() - t0

    print({"jit_compile_s": compile_s, "2_steps_s": step_s})
