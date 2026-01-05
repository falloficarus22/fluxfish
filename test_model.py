#!/usr/bin/env python3
"""
FluxFish Model Tester
Tests the trained LRT model on chess positions.
"""

import os
import sys

# Add repo to path
repo_root = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(repo_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import chess
from flax.training import train_state, checkpoints
import optax

from liquid_chess.models.lrt.complete_model import UltraFastLRT


def load_model(checkpoint_dir: str, config: dict):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_dir}...")
    
    model = UltraFastLRT(config)
    
    # Create dummy state for restoration
    dummy_input = {
        'pieces': jnp.zeros((8, 8), dtype=jnp.int8),
        'turn': jnp.array(True, dtype=jnp.bool_),
        'castling': jnp.zeros((4,), dtype=jnp.bool_),
        'ep_square': jnp.array(-1, dtype=jnp.int8)
    }
    
    rng = random.PRNGKey(0)
    params = model.init(rng, dummy_input)['params']
    
    # Create a dummy optimizer (needed for TrainState structure)
    tx = optax.adam(1e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # Restore checkpoint
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)
    print(f"✅ Loaded checkpoint from step {state.step}")
    
    return model, state.params


def board_to_input(board: chess.Board) -> dict:
    """Convert chess.Board to model input format."""
    pieces = np.zeros((8, 8), dtype=np.int8)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Map to index 1-12 (0 = empty)
            idx = (piece.piece_type - 1) * 2 + (1 if piece.color == chess.WHITE else 2)
            pieces[chess.square_rank(square), chess.square_file(square)] = idx
    
    return {
        'pieces': jnp.array(pieces),
        'turn': jnp.array(board.turn, dtype=jnp.bool_),
        'castling': jnp.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=jnp.bool_),
        'ep_square': jnp.array(
            board.ep_square if board.ep_square else -1,
            dtype=jnp.int8
        )
    }


def get_top_moves(board: chess.Board, policy: np.ndarray, top_k: int = 5):
    """Extract top-k legal moves from policy output."""
    legal_moves = list(board.legal_moves)
    
    move_probs = []
    for move in legal_moves:
        prob = policy[move.from_square, move.to_square]
        move_probs.append((move, float(prob)))
    
    # Sort by probability
    move_probs.sort(key=lambda x: x[1], reverse=True)
    
    return move_probs[:top_k]


def evaluate_position(model, params, board: chess.Board, verbose: bool = True):
    """Evaluate a single position."""
    board_input = board_to_input(board)
    
    # Run model
    output = model.apply({'params': params}, board_input, max_steps=32, deterministic=True)
    
    value = float(output['value'])
    policy = np.array(output['policy'])
    steps = int(output['stats']['steps_taken'])
    
    # Get top moves
    top_moves = get_top_moves(board, policy, top_k=5)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Position: {board.fen()}")
        print(f"Turn: {'White' if board.turn else 'Black'}")
        print(f"\nModel Evaluation: {value:+.3f} (normalized -1 to +1)")
        print(f"Reasoning Steps Used: {steps}")
        print(f"\nTop 5 Move Predictions:")
        for i, (move, prob) in enumerate(top_moves):
            san = board.san(move)
            print(f"  {i+1}. {san:8s} ({prob*100:5.2f}%)")
    
    return {
        'value': value,
        'policy': policy,
        'top_moves': top_moves,
        'steps': steps
    }


# Piece values for blunder detection
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # Can't be captured
}


def is_blunder(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if a move is an obvious blunder.
    A move is a blunder if:
    1. It hangs a piece (moves to a square attacked by opponent)
    2. It allows opponent to capture something valuable
    """
    # Make the move temporarily
    board.push(move)
    
    # Check if opponent can now capture something valuable
    dominated = False
    our_color = not board.turn  # We just moved, so it's opponent's turn
    
    # Find highest value piece opponent can capture
    best_capture_value = 0
    for opp_move in board.legal_moves:
        if board.is_capture(opp_move):
            captured = board.piece_at(opp_move.to_square)
            if captured:
                capture_value = PIECE_VALUES.get(captured.piece_type, 0)
                
                # Check if the capturing piece is protected
                attacker = board.piece_at(opp_move.from_square)
                attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
                
                # Net gain for opponent
                net_gain = capture_value - attacker_value
                if net_gain > 0 or capture_value >= 3:  # Losing a piece is bad
                    best_capture_value = max(best_capture_value, capture_value)
    
    # Check if the piece we moved is now hanging
    moved_piece = board.piece_at(move.to_square)
    if moved_piece:
        attackers = board.attackers(board.turn, move.to_square)
        defenders = board.attackers(our_color, move.to_square)
        
        if attackers and not defenders:
            # Piece is hanging!
            best_capture_value = max(best_capture_value, PIECE_VALUES.get(moved_piece.piece_type, 0))
    
    board.pop()
    
    # Consider it a blunder if we lose 3+ points of material
    return best_capture_value >= 3


def filter_blunders(board: chess.Board, top_moves: list) -> list:
    """Filter out obvious blunders from move list."""
    safe_moves = []
    for move, prob in top_moves:
        if not is_blunder(board, move):
            safe_moves.append((move, prob))
    
    # If all moves are blunders, return original list (we have to play something)
    if not safe_moves:
        return top_moves
    
    return safe_moves


def play_game(model, params, play_as_white: bool = False):
    """Play an interactive game against the model."""
    board = chess.Board()
    
    print("\n" + "="*50)
    print("♔ FluxFish Interactive Game ♚")
    print("="*50)
    print(f"You are playing as: {'White' if play_as_white else 'Black'}")
    print("Enter moves in UCI format (e.g., e2e4) or 'quit' to exit.\n")
    
    while not board.is_game_over():
        print(board)
        print()
        
        if board.turn == play_as_white:
            # Human's turn
            while True:
                move_str = input("Your move: ").strip()
                if move_str.lower() == 'quit':
                    print("Game aborted.")
                    return
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except:
                    print("Invalid format. Use UCI (e.g., e2e4).")
        else:
            # Model's turn
            print("FluxFish is thinking...")
            result = evaluate_position(model, params, board, verbose=False)
            
            if result['top_moves']:
                # Filter out obvious blunders
                safe_moves = filter_blunders(board, result['top_moves'])
                
                best_move, prob = safe_moves[0]
                san = board.san(best_move)
                
                # Indicate if we avoided a blunder
                if safe_moves != result['top_moves']:
                    orig_move = result['top_moves'][0][0]
                    print(f"⚠️  Avoided blunder: {board.san(orig_move)}")
                
                print(f"FluxFish plays: {san} (confidence: {prob*100:.1f}%)")
                print(f"Evaluation: {result['value']:+.3f}")
                board.push(best_move)
            else:
                print("No legal moves!")
                break
        
        print()
    
    print("\n" + "="*50)
    print(f"Game Over: {board.result()}")
    print(board)


def run_test_suite(model, params):
    """Run model on a set of test positions."""
    test_positions = [
        # Starting position
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position"),
        
        # After 1.e4
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
        
        # Italian Game
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Italian Game"),
        
        # Sicilian Defense
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "Sicilian Defense"),
        
        # Simple tactic: Fork
        ("r1bqkb1r/pppp1ppp/2n2n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 4 4", "Knight Fork Possible"),
        
        # Endgame: K+R vs K
        ("8/8/8/4k3/8/8/8/4K2R w - - 0 1", "Rook Endgame"),
    ]
    
    print("\n" + "="*60)
    print("🧪 FluxFish Test Suite")
    print("="*60)
    
    for fen, name in test_positions:
        print(f"\n📍 {name}")
        board = chess.Board(fen)
        evaluate_position(model, params, board, verbose=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FluxFish model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Model hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--play", action="store_true",
                        help="Play an interactive game")
    parser.add_argument("--play-white", action="store_true",
                        help="Play as white (default: black)")
    parser.add_argument("--fen", type=str, default=None,
                        help="Evaluate a specific FEN position")
    
    args = parser.parse_args()
    
    # Model config
    config = {
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'max_steps': 32,
        'min_reasoning_steps': 2,
        'dropout_rate': 0.0,
        'deterministic': True,
        'use_enhanced_encoder': False,
    }
    
    # Load model
    model, params = load_model(args.checkpoint_dir, config)
    
    if args.fen:
        # Evaluate specific position
        board = chess.Board(args.fen)
        evaluate_position(model, params, board)
    elif args.play:
        # Interactive game
        play_game(model, params, play_as_white=args.play_white)
    else:
        # Run test suite
        run_test_suite(model, params)


if __name__ == "__main__":
    main()
