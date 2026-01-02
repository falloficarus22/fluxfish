import pytest
import chess
from pathlib import Path
import tempfile
import os


def _add_repo_python_to_path():
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def _write_test_pgn(path: Path):
    """Write a small test PGN file"""
    chess_pgn = pytest.importorskip("chess.pgn")
    
    game = chess_pgn.Game()
    board = chess.Board()
    node = game
    
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        board.push(move)
        node = node.add_variation(move)
    
    game.headers["Result"] = "1/2-1/2"
    
    with path.open("w") as f:
        print(game, file=f)


def test_enhanced_encoder_with_model():
    """Test that enhanced encoder integrates with model"""
    _add_repo_python_to_path()
    
    import jax
    from jax import random
    from liquid_chess.models.lrt.complete_model import UltraFastLRT
    from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input
    
    # Config with enhanced encoder
    config = {
        "hidden_dim": 128,
        "num_heads": 4,
        "max_steps": 4,
        "use_enhanced_encoder": True,
        "dropout_rate": 0.0,
        "deterministic": True,
    }
    
    model = UltraFastLRT(config)
    
    # Create input
    board = chess.Board()
    board_input = board_to_enhanced_input(board)
    
    # Initialize model
    rng = random.PRNGKey(0)
    params = model.init(rng, board_input, max_steps=4, deterministic=True)
    
    # Forward pass
    outputs = model.apply(
        params,
        board_input,
        max_steps=4,
        deterministic=True
    )
    
    assert 'value' in outputs
    assert 'policy' in outputs
    assert not jax.numpy.isnan(outputs['value']).any()


def test_enhanced_dataset_loading():
    """Test dataset loads with enhanced features"""
    _add_repo_python_to_path()
    
    from liquid_chess.training.trainer import ChessDataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pgn_path = Path(tmpdir) / "test.pgn"
        _write_test_pgn(pgn_path)
        
        # Load with enhanced features
        dataset = ChessDataset(
            [str(pgn_path)],
            batch_size=2,
            use_enhanced_features=True
        )
        
        assert len(dataset.positions) > 0
        
        # Check first example has all enhanced features
        example = dataset.positions[0]
        
        enhanced_keys = [
            'white_attacks', 'black_attacks', 'pins',
            'king_safety_white', 'king_safety_black',
            'passed_pawns', 'isolated_pawns', 'doubled_pawns',
            'material_imbalance'
        ]
        
        for key in enhanced_keys:
            assert key in example, f"Missing enhanced feature: {key}"


def test_training_step_with_enhanced_encoder():
    """Test that training step works with enhanced encoder"""
    _add_repo_python_to_path()
    
    import jax
    from jax import random
    from liquid_chess.training.trainer import ChessDataset, LRTTrainer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pgn_path = Path(tmpdir) / "test.pgn"
        _write_test_pgn(pgn_path)
        
        config = {
            "seed": 0,
            "model": {
                "hidden_dim": 64,
                "num_heads": 2,
                "max_steps": 2,
                "dropout_rate": 0.0,
                "use_enhanced_encoder": True,
                "deterministic": True,
            },
            "training": {
                "learning_rate": 1e-3,
                "end_learning_rate": 1e-4,
                "warmup_steps": 0,
                "total_steps": 2,
                "steps_per_epoch": 1,
                "val_steps": 1,
                "batch_size": 2,
                "max_grad_norm": 1.0,
                "weight_decay": 0.0,
                "value_weight": 1.0,
                "policy_weight": 1.0,
                "step_penalty": 0.0,
                "checkpoint_dir": str(Path(tmpdir) / "ckpt"),
                "save_every": 1000,
                "keep_checkpoints": 1,
            },
            "logging": {"use_wandb": False},
        }
        
        dataset = ChessDataset(
            [str(pgn_path)],
            batch_size=2,
            use_enhanced_features=True
        )
        
        trainer = LRTTrainer(config)
        trainer.state = trainer.create_train_state()
        
        batch = dataset.get_batch()
        rng = random.PRNGKey(0)
        
        # This should work without errors
        new_state, metrics = trainer.train_step(trainer.state, batch, rng)
        
        assert 'loss' in metrics
        assert not jax.numpy.isnan(metrics['loss']).any()


def test_comparison_simple_vs_enhanced():
    """Compare simple and enhanced encoders"""
    _add_repo_python_to_path()
    
    import jax
    from jax import random
    from liquid_chess.models.lrt.complete_model import UltraFastLRT
    from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input
    
    board = chess.Board()
    
    # Test both encoders
    configs = [
        {"hidden_dim": 128, "num_heads": 4, "max_steps": 4, 
         "use_enhanced_encoder": False, "dropout_rate": 0.0, "deterministic": True},
        {"hidden_dim": 128, "num_heads": 4, "max_steps": 4, 
         "use_enhanced_encoder": True, "dropout_rate": 0.0, "deterministic": True},
    ]
    
    outputs_list = []
    
    for config in configs:
        model = UltraFastLRT(config)
        
        # Create appropriate input
        if config["use_enhanced_encoder"]:
            board_input = board_to_enhanced_input(board)
        else:
            import jax.numpy as jnp
            pieces = jnp.zeros((8, 8), dtype=jnp.int8)
            board_input = {
                'pieces': pieces,
                'turn': jnp.array(True, dtype=jnp.bool_),
                'castling': jnp.zeros(4, dtype=jnp.bool_),
                'ep_square': jnp.array(-1, dtype=jnp.int8),
            }
        
        rng = random.PRNGKey(42)
        params = model.init(rng, board_input, max_steps=4, deterministic=True)
        outputs = model.apply(params, board_input, max_steps=4, deterministic=True)
        outputs_list.append(outputs)
    
    # Both should produce valid outputs
    for outputs in outputs_list:
        assert 'value' in outputs
        assert 'policy' in outputs
        assert not jax.numpy.isnan(outputs['value']).any()