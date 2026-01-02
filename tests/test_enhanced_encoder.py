import pytest
import chess
import jax
from jax import random
import jax.numpy as jnp
from liquid_chess.models.lrt.enhanced_encoder import EnhancedChessBoardEncoder
from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input


def test_encoder_initialization():
    """Test encoder can be initialized"""
    encoder = EnhancedChessBoardEncoder(features=512)
    assert encoder.features == 512


def test_encoder_forward_pass():
    """Test encoder forward pass"""
    encoder = EnhancedChessBoardEncoder(features=256)
    
    # Create sample input
    board = chess.Board()
    board_input = board_to_enhanced_input(board)
    
    # Initialize encoder
    rng = random.PRNGKey(0)
    params = encoder.init(rng, board_input)
    
    # Forward pass
    encoded = encoder.apply(params, board_input)
    
    # Check output shape
    assert encoded.shape == (64, 256)
    
    # Check no NaN values
    assert not jnp.isnan(encoded).any()


def test_encoder_different_positions():
    """Test encoder produces different outputs for different positions"""
    encoder = EnhancedChessBoardEncoder(features=128)
    rng = random.PRNGKey(42)
    
    # Position 1: Starting position
    board1 = chess.Board()
    input1 = board_to_enhanced_input(board1)
    
    # Position 2: After e4
    board2 = chess.Board()
    board2.push_san("e4")
    input2 = board_to_enhanced_input(board2)
    
    # Initialize with first position
    params = encoder.init(rng, input1)
    
    # Encode both positions
    encoded1 = encoder.apply(params, input1)
    encoded2 = encoder.apply(params, input2)
    
    # Outputs should be different
    diff = jnp.sum(jnp.abs(encoded1 - encoded2))
    assert diff > 1.0, "Encoder produces same output for different positions"


def test_encoder_batch_compatibility():
    """Test encoder works with batched inputs"""
    encoder = EnhancedChessBoardEncoder(features=128)
    rng = random.PRNGKey(0)
    
    # Create batch of 4 positions
    boards = [chess.Board() for _ in range(4)]
    for i, board in enumerate(boards[1:], 1):
        for _ in range(i):
            moves = list(board.legal_moves)
            if moves:
                board.push(moves[0])
    
    # Stack inputs
    inputs = [board_to_enhanced_input(b) for b in boards]
    
    # Stack each feature
    batched_input = {}
    for key in inputs[0].keys():
        batched_input[key] = jnp.stack([inp[key] for inp in inputs])
    
    # Initialize
    params = encoder.init(rng, inputs[0])
    
    # Process batch with vmap
    from jax import vmap
    encode_batch = vmap(lambda x: encoder.apply(params, x))
    
    # This should work without errors
    encoded_batch = encode_batch(batched_input)
    
    assert encoded_batch.shape == (4, 64, 128)


def test_encoder_gradient_flow():
    """Test that gradients flow through encoder"""
    encoder = EnhancedChessBoardEncoder(features=64)
    rng = random.PRNGKey(0)
    
    board = chess.Board()
    board_input = board_to_enhanced_input(board)
    
    params = encoder.init(rng, board_input)
    
    # Define a simple loss function
    def loss_fn(params):
        encoded = encoder.apply(params, board_input)
        return jnp.sum(encoded ** 2)
    
    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)
    
    # Check gradients are non-zero
    grad_values = jax.tree_util.tree_leaves(grads)
    total_grad = sum(jnp.sum(jnp.abs(g)) for g in grad_values)
    
    assert total_grad > 0, "No gradients flowing through encoder"


@pytest.mark.parametrize("features", [64, 128, 256, 512])
def test_encoder_different_sizes(features):
    """Test encoder works with different feature sizes"""
    encoder = EnhancedChessBoardEncoder(features=features)
    rng = random.PRNGKey(0)
    
    board = chess.Board()
    board_input = board_to_enhanced_input(board)
    
    params = encoder.init(rng, board_input)
    encoded = encoder.apply(params, board_input)
    
    assert encoded.shape == (64, features)