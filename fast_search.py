#!/usr/bin/env python3
"""
Fast MCTS for FluxFish LRT - Optimized for CPU
Uses batched evaluation and aggressive pruning for speed.
"""

import os
import sys
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add repo to path
repo_root = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(repo_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

import numpy as np
import chess
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax

from liquid_chess.models.lrt.complete_model import UltraFastLRT


class FastMCTS:
    """
    Lightweight MCTS optimized for CPU inference.
    Uses aggressive pruning and caching for speed.
    """
    
    def __init__(self, model: UltraFastLRT, params: dict):
        self.model = model
        self.params = params
        self.eval_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        
        # JIT-compile the evaluation function for speed
        self._init_jit_eval()
    
    def _init_jit_eval(self):
        """Create JIT-compiled evaluation function."""
        @jax.jit
        def _eval_fn(params, board_input):
            return self.model.apply(
                {'params': params},
                board_input,
                max_steps=16,  # Reduced steps for speed
                deterministic=True
            )
        self._jit_eval = _eval_fn
    
    def board_to_input(self, board: chess.Board) -> dict:
        """Convert board to model input."""
        pieces = np.zeros((8, 8), dtype=np.int8)
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc:
                idx = (pc.piece_type - 1) * 2 + (1 if pc.color == chess.WHITE else 2)
                pieces[chess.square_rank(sq), chess.square_file(sq)] = idx
        
        return {
            'pieces': jnp.array(pieces),
            'turn': jnp.array(board.turn, dtype=jnp.bool_),
            'castling': jnp.array([
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)
            ], dtype=jnp.bool_),
            'ep_square': jnp.array(board.ep_square if board.ep_square else -1, dtype=jnp.int8)
        }
    
    def evaluate(self, board: chess.Board) -> Tuple[float, np.ndarray]:
        """Get evaluation with caching."""
        fen = board.fen()
        if fen in self.eval_cache:
            return self.eval_cache[fen]
        
        board_input = self.board_to_input(board)
        output = self._jit_eval(self.params, board_input)
        
        value = float(output['value'])
        policy = np.array(output['policy'])
        
        self.eval_cache[fen] = (value, policy)
        return value, policy
    
    def get_move_priors(self, board: chess.Board, policy: np.ndarray) -> List[Tuple[chess.Move, float]]:
        """Extract move priors from policy, sorted by probability."""
        moves_priors = []
        for move in board.legal_moves:
            prior = policy[move.from_square, move.to_square]
            moves_priors.append((move, float(prior)))
        
        # Sort by prior descending
        moves_priors.sort(key=lambda x: x[1], reverse=True)
        return moves_priors
    
    def minimax_search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float = -float('inf'),
        beta: float = float('inf'),
        use_nn: bool = True
    ) -> Tuple[float, Optional[chess.Move]]:
        """
        Simple alpha-beta search with NN evaluation at leaves.
        Much faster than full MCTS for shallow depths.
        """
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return (100.0, None) if board.turn == chess.WHITE else (-100.0, None)
            elif result == "0-1":
                return (-100.0, None) if board.turn == chess.WHITE else (100.0, None)
            return (0.0, None)
        
        if depth <= 0:
            if use_nn:
                value, _ = self.evaluate(board)
                # Flip value for black's perspective
                return (value if board.turn == chess.WHITE else -value, None)
            else:
                return (0.0, None)
        
        # Get move ordering from NN policy
        _, policy = self.evaluate(board)
        moves_priors = self.get_move_priors(board, policy)
        
        # Only consider top moves for speed (aggressive pruning)
        top_k = min(8, len(moves_priors))  # Only look at top 8 moves
        moves_priors = moves_priors[:top_k]
        
        best_value = -float('inf')
        best_move = moves_priors[0][0] if moves_priors else None
        
        for move, prior in moves_priors:
            board.push(move)
            child_value, _ = self.minimax_search(board, depth - 1, -beta, -alpha, use_nn)
            child_value = -child_value  # Negamax
            board.pop()
            
            if child_value > best_value:
                best_value = child_value
                best_move = move
            
            alpha = max(alpha, child_value)
            if alpha >= beta:
                break  # Beta cutoff
        
        return (best_value, best_move)
    
    def search(
        self,
        board: chess.Board,
        depth: int = 3,
        time_limit: float = None
    ) -> Tuple[chess.Move, Dict]:
        """
        Search for best move using NN-guided alpha-beta.
        
        Args:
            board: Current position
            depth: Search depth (2-4 recommended for speed)
            time_limit: Optional time limit in seconds
        """
        start = time.time()
        
        value, best_move = self.minimax_search(board, depth)
        
        elapsed = time.time() - start
        
        # Get NN evaluation for stats
        nn_value, policy = self.evaluate(board)
        
        stats = {
            'depth': depth,
            'value': value,
            'nn_value': nn_value,
            'time': elapsed,
            'cache_size': len(self.eval_cache),
        }
        
        return best_move, stats


def load_model(checkpoint_dir: str, config: dict):
    """Load trained model."""
    print(f"Loading model from {checkpoint_dir}...")
    
    model = UltraFastLRT(config)
    
    dummy_input = {
        'pieces': jnp.zeros((8, 8), dtype=jnp.int8),
        'turn': jnp.array(True, dtype=jnp.bool_),
        'castling': jnp.zeros((4,), dtype=jnp.bool_),
        'ep_square': jnp.array(-1, dtype=jnp.int8)
    }
    
    rng = random.PRNGKey(0)
    params = model.init(rng, dummy_input)['params']
    
    tx = optax.adam(1e-4)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)
    print(f"✅ Loaded checkpoint from step {state.step}")
    
    return model, state.params


def play_game(model, params, depth: int = 3, play_as_white: bool = False):
    """Play an interactive game."""
    board = chess.Board()
    search = FastMCTS(model, params)
    
    print("\n" + "="*50)
    print("♔ FluxFish Fast Search ♚")
    print("="*50)
    print(f"Search depth: {depth}")
    print(f"You are: {'White' if play_as_white else 'Black'}")
    print("Enter moves in UCI (e.g., e2e4) or 'quit'\n")
    
    while not board.is_game_over():
        print(board)
        print()
        
        if board.turn == play_as_white:
            # Human turn
            while True:
                move_str = input("Your move: ").strip()
                if move_str == 'quit':
                    return
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    print("Illegal move.")
                except:
                    print("Invalid format.")
        else:
            # Engine turn
            print("Thinking...")
            move, stats = search.search(board, depth=depth)
            
            if move:
                san = board.san(move)
                print(f"\nFluxFish plays: {san}")
                print(f"  Eval: {stats['value']:+.2f} (NN: {stats['nn_value']:+.2f})")
                print(f"  Time: {stats['time']:.2f}s")
                print(f"  Cache: {stats['cache_size']} positions")
                board.push(move)
        print()
    
    print(f"\nGame Over: {board.result()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FluxFish Fast Search")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3, help="Search depth (2-4)")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-white", action="store_true")
    parser.add_argument("--fen", type=str, default=None)
    
    args = parser.parse_args()
    
    config = {
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'max_steps': 16,  # Reduced for speed
        'min_reasoning_steps': 2,
        'dropout_rate': 0.0,
        'deterministic': True,
        'use_enhanced_encoder': False,
    }
    
    model, params = load_model(args.checkpoint_dir, config)
    
    if args.fen:
        board = chess.Board(args.fen)
        search = FastMCTS(model, params)
        print(f"\nAnalyzing: {args.fen}")
        print(board)
        move, stats = search.search(board, depth=args.depth)
        if move:
            print(f"\nBest: {board.san(move)}")
            print(f"Eval: {stats['value']:+.2f}")
    elif args.play:
        play_game(model, params, args.depth, args.play_white)
    else:
        # Quick benchmark
        board = chess.Board()
        search = FastMCTS(model, params)
        print("\nBenchmarking on starting position...")
        for d in [2, 3, 4]:
            move, stats = search.search(board, depth=d)
            print(f"Depth {d}: {board.san(move)} ({stats['time']:.2f}s)")


if __name__ == "__main__":
    main()
