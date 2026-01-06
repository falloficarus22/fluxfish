#!/usr/bin/env python3
"""
MCTS Search for FluxFish LRT
Monte Carlo Tree Search using the trained LRT model for evaluation and policy priors.
"""

import os
import sys
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add repo to path
repo_root = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(repo_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

import numpy as np
import chess
import jax.numpy as jnp
from jax import random
from flax.training import train_state, checkpoints
import optax

from liquid_chess.models.lrt.complete_model import UltraFastLRT


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    board: chess.Board
    parent: Optional['MCTSNode'] = None
    move: Optional[chess.Move] = None  # Move that led to this node
    children: Dict[chess.Move, 'MCTSNode'] = field(default_factory=dict)
    
    # MCTS statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # LRT outputs (cached)
    policy: Optional[np.ndarray] = None
    lrt_value: Optional[float] = None
    is_expanded: bool = False
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Upper Confidence Bound score for selection."""
        if self.parent is None:
            return 0.0
        
        # Exploration term
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        # Exploitation term (negative because we want opponent's bad moves)
        exploitation = -self.value if self.visit_count > 0 else 0.0
        
        return exploitation + exploration


class MCTS:
    """
    Monte Carlo Tree Search with LRT neural network guidance.
    
    Uses the LRT model to:
    1. Provide move priors (policy head)
    2. Evaluate leaf nodes (value head)
    """
    
    def __init__(
        self,
        model: UltraFastLRT,
        params: dict,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        batch_size: int = 8  # Evaluate this many positions at once
    ):
        self.model = model
        self.params = params
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.batch_size = batch_size
        
        # Statistics
        self.nodes_created = 0
        self.cache_hits = 0
        
        # Position cache (avoid re-evaluating same positions)
        self.eval_cache: Dict[str, Tuple[float, np.ndarray]] = {}
        
        # Pending evaluations for batching
        self.pending_nodes: List[MCTSNode] = []
    
    def board_to_input(self, board: chess.Board) -> dict:
        """Convert chess.Board to model input format using enhanced features."""
        from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input
        return board_to_enhanced_input(board)
    
    def batch_evaluate(self, boards: List[chess.Board]) -> List[Tuple[float, np.ndarray]]:
        """Evaluate multiple positions at once (highly optimized)."""
        results = [None] * len(boards)
        uncached_boards = []
        uncached_indices = []
        
        # 1. Check cache (important for transposition speed)
        for i, board in enumerate(boards):
            fen = board.fen()
            if fen in self.eval_cache:
                self.cache_hits += 1
                results[i] = self.eval_cache[fen]
            else:
                uncached_boards.append(board)
                uncached_indices.append(i)
        
        # 2. Batch evaluate uncached positions
        if uncached_boards:
            # Gather enhanced inputs
            inputs_list = [self.board_to_input(b) for b in uncached_boards]
            batch_inputs = {
                key: jnp.stack([inp[key] for inp in inputs_list])
                for key in inputs_list[0].keys()
            }
            
            # Use cached JIT function if available for 10x speedup
            if not hasattr(self, '_jit_eval_fn'):
                def single_eval(params, board_input):
                    return self.model.apply(
                        {'params': params},
                        board_input,
                        max_steps=8, # Fast search depth
                        deterministic=True
                    )
                self._jit_eval_fn = jax.jit(jax.vmap(single_eval, in_axes=(None, 0)))
            
            outputs = self._jit_eval_fn(self.params, batch_inputs)
            
            # Unpack results and update cache
            for i, idx in enumerate(uncached_indices):
                val = float(outputs['value'][i])
                pol = np.array(outputs['policy'][i])
                res = (val, pol)
                self.eval_cache[uncached_boards[i].fen()] = res
                results[idx] = res
                
        return results
    
    def evaluate(self, board: chess.Board) -> Tuple[float, np.ndarray]:
        """Get LRT evaluation for a position."""
        fen = board.fen()
        
        # Check cache
        if fen in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[fen]
        
        # Run model
        board_input = self.board_to_input(board)
        output = self.model.apply(
            {'params': self.params}, 
            board_input, 
            max_steps=32, 
            deterministic=True
        )
        
        value = float(output['value'])
        policy = np.array(output['policy'])
        
        # Cache result
        self.eval_cache[fen] = (value, policy)
        
        return value, policy
    
    def expand(self, node: MCTSNode) -> None:
        """Expand a node by adding all legal moves as children."""
        if node.is_expanded or node.board.is_game_over():
            return
        
        # Get LRT evaluation
        value, policy = self.evaluate(node.board)
        node.lrt_value = value
        node.policy = policy
        
        # Create children for all legal moves
        legal_moves = list(node.board.legal_moves)
        
        # Extract priors from policy
        priors = []
        for move in legal_moves:
            prior = policy[move.from_square, move.to_square]
            priors.append(prior)
        
        # Normalize priors
        prior_sum = sum(priors)
        if prior_sum > 0:
            priors = [p / prior_sum for p in priors]
        else:
            priors = [1.0 / len(legal_moves)] * len(legal_moves)
        
        # Add Dirichlet noise to root node for exploration
        if node.parent is None and self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            priors = [
                (1 - self.dirichlet_epsilon) * p + self.dirichlet_epsilon * n
                for p, n in zip(priors, noise)
            ]
        
        # Create child nodes
        for move, prior in zip(legal_moves, priors):
            child_board = node.board.copy()
            child_board.push(move)
            
            child = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior=prior
            )
            node.children[move] = child
            self.nodes_created += 1
        
        node.is_expanded = True
    
    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using UCB."""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def simulate(self, node: MCTSNode) -> float:
        """Run one MCTS simulation from the given node."""
        path = [node]
        
        # Selection: traverse tree using UCB
        current = node
        while current.is_expanded and current.children and not current.board.is_game_over():
            current = self.select_child(current)
            path.append(current)
        
        # Expansion & Evaluation
        if not current.is_expanded and not current.board.is_game_over():
            self.expand(current)
        
        # Get value for backpropagation
        if current.board.is_game_over():
            # Terminal node: use actual game result
            result = current.board.result()
            if result == "1-0":
                value = 1.0 if current.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = 1.0 if current.board.turn == chess.WHITE else -1.0
            else:
                value = 0.0
        else:
            # Use LRT value
            value = current.lrt_value if current.lrt_value is not None else 0.0
        
        # Backpropagation: update all nodes in path
        for i, node_in_path in enumerate(reversed(path)):
            # Flip value for alternating players
            sign = 1 if i % 2 == 0 else -1
            node_in_path.visit_count += 1
            node_in_path.value_sum += sign * value
        
        return value
    
    def search(
        self, 
        board: chess.Board, 
        num_simulations: int = 800,
        time_limit: float = None
    ) -> Tuple[chess.Move, Dict]:
        """
        Run MCTS search and return the best move.
        
        Args:
            board: Current position
            num_simulations: Number of MCTS simulations
            time_limit: Optional time limit in seconds
        
        Returns:
            best_move: The selected best move
            stats: Search statistics
        """
        start_time = time.time()
        
        # Create root node
        root = MCTSNode(board=board.copy())
        self.expand(root)
        
        # Run simulations
        simulations_done = 0
        while simulations_done < num_simulations:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            self.simulate(root)
            simulations_done += 1
        
        elapsed = time.time() - start_time
        
        # Select best move based on visit counts
        if not root.children:
            return None, {}
        
        # Apply temperature to visit counts
        visits = np.array([child.visit_count for child in root.children.values()])
        moves = list(root.children.keys())
        
        if self.temperature == 0:
            # Deterministic: pick most visited
            best_idx = np.argmax(visits)
        else:
            # Sample based on visit counts ^ (1/temp)
            visit_probs = visits ** (1.0 / self.temperature)
            visit_probs = visit_probs / visit_probs.sum()
            best_idx = np.random.choice(len(moves), p=visit_probs)
        
        best_move = moves[best_idx]
        best_child = root.children[best_move]
        
        # Policy distribution (normalized visit counts)
        policy_probs = visits / visits.sum()
        policy_dist = np.zeros((64, 64), dtype=np.float32)
        for move, prob in zip(moves, policy_probs):
            policy_dist[move.from_square, move.to_square] = prob
            
        # Collect statistics
        stats = {
            'simulations': simulations_done,
            'time': elapsed,
            'nps': simulations_done / elapsed if elapsed > 0 else 0,
            'root_value': root.lrt_value,
            'best_value': -best_child.value,  # Flip sign (opponent's perspective)
            'best_visits': best_child.visit_count,
            'nodes_created': self.nodes_created,
            'cache_hits': self.cache_hits,
            'pv': self.get_pv(root),
            'policy_dist': policy_dist,
        }
        
        return best_move, stats
    
    def get_pv(self, node: MCTSNode, max_depth: int = 10) -> List[chess.Move]:
        """Extract the principal variation (most visited path)."""
        pv = []
        current = node
        
        for _ in range(max_depth):
            if not current.children:
                break
            
            # Pick most visited child
            best_child = max(current.children.values(), key=lambda c: c.visit_count)
            pv.append(best_child.move)
            current = best_child
        
        return pv


def load_model(checkpoint_dir: str, config: dict):
    """Load trained model from checkpoint."""
    if checkpoint_dir is None:
        print("⚠️ No checkpoint provided, initializing random model.")
        model = UltraFastLRT(config)
        rng = random.PRNGKey(0)
        dummy_input = {
            'pieces': jnp.zeros((8, 8, 12), dtype=jnp.float32),
            'turn': jnp.array(True, dtype=jnp.bool_),
            'castling': jnp.zeros((4,), dtype=jnp.bool_),
            'ep_square': jnp.array(-1, dtype=jnp.int8)
        }
        params = model.init(rng, dummy_input)['params']
        return model, params

    print(f"Loading model from {checkpoint_dir}...")
    model = UltraFastLRT(config)
    
    # Dummy input for initialization
    dummy_input = {
        'pieces': jnp.zeros((8, 8, 12), dtype=jnp.float32),
        'turn': jnp.array(True, dtype=jnp.bool_),
        'castling': jnp.zeros((4,), dtype=jnp.bool_),
        'ep_square': jnp.array(-1, dtype=jnp.int8)
    }
    
    rng = random.PRNGKey(0)
    params = model.init(rng, dummy_input)['params']
    
    tx = optax.adam(1e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # Support both absolute and relative paths
    abs_path = os.path.abspath(checkpoint_dir)
    if os.path.exists(abs_path):
        state = checkpoints.restore_checkpoint(ckpt_dir=abs_path, target=state)
        print(f"✅ Loaded checkpoint from step {state.step}")
    else:
        print(f"⚠️ Checkpoint path {abs_path} not found, using random weights.")
    
    return model, state.params


def play_game_mcts(model, params, num_simulations: int = 200, play_as_white: bool = False):
    """Play an interactive game using MCTS search."""
    board = chess.Board()
    mcts = MCTS(model, params, temperature=0.1)  # Low temp for strong play
    
    print("\n" + "="*60)
    print("♔ FluxFish MCTS Interactive Game ♚")
    print("="*60)
    print(f"Simulations per move: {num_simulations}")
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
            # MCTS search
            print(f"FluxFish is thinking ({num_simulations} simulations)...")
            
            move, stats = mcts.search(board, num_simulations=num_simulations)
            
            if move:
                san = board.san(move)
                print(f"\nFluxFish plays: {san}")
                print(f"  Value: {stats['best_value']:+.3f}")
                print(f"  Visits: {stats['best_visits']}")
                print(f"  Time: {stats['time']:.2f}s ({stats['nps']:.0f} sim/s)")
                
                # Show PV
                if stats['pv']:
                    pv_board = board.copy()
                    pv_san = []
                    for pv_move in stats['pv'][:5]:
                        pv_san.append(pv_board.san(pv_move))
                        pv_board.push(pv_move)
                    print(f"  PV: {' '.join(pv_san)}")
                
                board.push(move)
            else:
                print("No legal moves!")
                break
        
        print()
    
    print("\n" + "="*60)
    print(f"Game Over: {board.result()}")
    print(board)


def analyze_position(model, params, fen: str, num_simulations: int = 800):
    """Analyze a position with MCTS."""
    board = chess.Board(fen)
    mcts = MCTS(model, params, temperature=0)
    
    print(f"\nAnalyzing: {fen}")
    print(board)
    print()
    
    move, stats = mcts.search(board, num_simulations=num_simulations)
    
    if move:
        san = board.san(move)
        print(f"Best Move: {san}")
        print(f"Value: {stats['best_value']:+.3f}")
        print(f"Simulations: {stats['simulations']}")
        print(f"Time: {stats['time']:.2f}s")
        
        if stats['pv']:
            pv_board = board.copy()
            pv_san = []
            for pv_move in stats['pv']:
                pv_san.append(pv_board.san(pv_move))
                pv_board.push(pv_move)
            print(f"PV: {' '.join(pv_san)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FluxFish MCTS Search")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move")
    parser.add_argument("--play", action="store_true",
                        help="Play an interactive game")
    parser.add_argument("--play-white", action="store_true",
                        help="Play as white")
    parser.add_argument("--fen", type=str, default=None,
                        help="Analyze a specific position")
    
    args = parser.parse_args()
    
    config = {
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'max_steps': 32,
        'min_reasoning_steps': 2,
        'dropout_rate': 0.0,
        'deterministic': True,
        'use_enhanced_encoder': False,
    }
    
    model, params = load_model(args.checkpoint_dir, config)
    
    if args.fen:
        analyze_position(model, params, args.fen, args.simulations)
    elif args.play:
        play_game_mcts(model, params, args.simulations, args.play_white)
    else:
        # Demo: analyze starting position
        analyze_position(
            model, params,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            args.simulations
        )


if __name__ == "__main__":
    main()
