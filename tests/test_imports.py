"""
test_imports.py - Basic tests to verify imports work correctly
"""

import pytest


def test_import_liquid_chess():
    """Test that we can import the main package"""
    import liquid_chess
    assert liquid_chess is not None


def test_import_existing_lrt_model():
    """Test that existing LRT model imports"""
    from liquid_chess.models.lrt import UltraFastLRT
    assert UltraFastLRT is not None


def test_import_chess():
    """Test that python-chess is available"""
    import chess
    board = chess.Board()
    assert board is not None
    assert len(list(board.legal_moves)) == 20  # Starting position has 20 legal moves


def test_import_jax():
    """Test that JAX is available"""
    import jax
    import jax.numpy as jnp
    
    x = jnp.array([1.0, 2.0, 3.0])
    assert x.shape == (3,)


def test_import_flax():
    """Test that Flax is available"""
    import flax.linen as nn
    
    class SimpleModule(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(10)(x)
    
    assert SimpleModule is not None


def test_feature_extraction_imports():
    """Test that feature extraction module can be imported"""
    try:
        from liquid_chess.models.lrt.feature_extraction import (
            ChessFeatureExtractor,
            board_to_enhanced_input
        )
        assert ChessFeatureExtractor is not None
        assert board_to_enhanced_input is not None
    except ImportError as e:
        pytest.skip(f"feature_extraction.py not yet created: {e}")


def test_enhanced_encoder_imports():
    """Test that enhanced encoder can be imported"""
    try:
        from liquid_chess.models.lrt.enhanced_encoder import EnhancedChessBoardEncoder
        assert EnhancedChessBoardEncoder is not None
    except ImportError as e:
        pytest.skip(f"enhanced_encoder.py not yet created: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])