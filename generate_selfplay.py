#!/usr/bin/env python3
import os
import sys
import time

# Add repo/python to sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if os.path.join(repo_root, "python") not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "python"))
import uuid
import chess
import chess.pgn
import chess.engine
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Dict
from liquid_chess.models.lrt.complete_model import UltraFastLRT
from mcts_search import MCTS, load_model

def play_self_game_engine(engine_path: str, num_simulations: int = 800) -> List[Dict]:
    """Play a game using the C++ UCI engine."""
    board = chess.Board()
    game_data = []
    
    # Start engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    while not board.is_game_over():
        # Set node limit for MCTS simulations
        result = engine.play(board, chess.engine.Limit(nodes=num_simulations))
        move = result.move
        
        if move is None:
            break
            
        # For RL we really want the search distribution (pi)
        # But UCI 'go' command usually only returns bestmove.
        # Ideally we'd parse 'info string' or similar if the engine outputs visit counts.
        # For now, we'll use a one-hot policy if distribution isn't available.
        policy = np.zeros((64, 64), dtype=np.float32)
        policy[move.from_square, move.to_square] = 1.0
        
        game_data.append({
            'fen': board.fen(),
            'policy': policy,
            'turn': board.turn
        })
        
        board.push(move)
    
    engine.quit()
    
    # Determine winner
    res_str = board.result()
    if res_str == "1-0": winner = chess.WHITE
    elif res_str == "0-1": winner = chess.BLACK
    else: winner = None
    
    final_data = []
    for entry in game_data:
        outcome = 0.0 if winner is None else (1.0 if entry['turn'] == winner else -1.0)
        final_data.append({
            'fen': entry['fen'],
            'policy': entry['policy'],
            'outcome': outcome
        })
    return final_data


def get_empty_model_params(config):
    """Initialize a model with random weights for the start of RL."""
    model = UltraFastLRT(config)
    dummy_input = {
        'pieces': jnp.zeros((8, 8), dtype=jnp.int8),
        'turn': jnp.array(True, dtype=jnp.bool_),
        'castling': jnp.zeros((4,), dtype=jnp.bool_),
        'ep_square': jnp.array(-1, dtype=jnp.int8)
    }
    import jax
    rng = jax.random.PRNGKey(int(time.time()))
    return model, model.init(rng, dummy_input)['params']

def play_self_game(mcts: MCTS, temp_threshold: int = 15) -> List[Dict]:
    """Play a single game against itself and return the states, search policies, and result."""
    board = chess.Board()
    game_data = []
    
    while not board.is_game_over():
        # Temperature control: use high temp for exploration early, then low temp for competitive play
        mcts.temperature = 1.0 if len(board.move_stack) < temp_threshold else 0.1
        
        move, stats = mcts.search(board, num_simulations=mcts.num_simulations)
        
        if move is None:
            break
            
        # Store state (FEN), search policy (visit counts), and meta
        # For policy target, we use normalized visit counts
        policy = np.zeros((64, 64), dtype=np.float32)
        # This is a bit slow - in a real MCTS stats['root_visits'] would be better
        # but let's use the stats we have or modify MCTS to return them.
        # stats['pv'] is there, but not full visit distribution.
        
        # Use the full search distribution as the policy target
        policy = stats.get('policy_dist', np.zeros((64, 64), dtype=np.float32))
        
        game_data.append({
            'fen': board.fen(),
            'policy': policy,
            'turn': board.turn
        })
        
        board.push(move)
        
    # Determine winner
    result = board.result()
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        winner = None # Draw
        
    # Annotate game data with normalized outcome
    final_data = []
    for entry in game_data:
        if winner is None:
            outcome = 0.0
        else:
            outcome = 1.0 if entry['turn'] == winner else -1.0
        
        final_data.append({
            'fen': entry['fen'],
            'policy': entry['policy'],
            'outcome': outcome
        })
        
    return final_data

def save_games(games: List[List[Dict]], output_dir: str):
    """Save generated games to a .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    
    fens = []
    policies = []
    outcomes = []
    
    for game in games:
        for move_data in game:
            fens.append(move_data['fen'])
            policies.append(move_data['policy'])
            outcomes.append(move_data['outcome'])
            
    timestamp = int(time.time())
    game_id = uuid.uuid4().hex[:8]
    filename = os.path.join(output_dir, f"selfplay_{timestamp}_{game_id}.npz")
    
    np.savez_compressed(
        filename,
        fens=np.array(fens),
        policies=np.array(policies),
        outcomes=np.array(outcomes)
    )
    print(f"Saved {len(fens)} positions to {filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/selfplay_rl")
    parser.add_argument("--engine", type=str, default=None, help="Path to C++ UCI engine")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    
    args = parser.parse_args()
    
    config = {
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'max_steps': 32,
        'min_reasoning_steps': 2,
        'dropout_rate': 0.0,
        'deterministic': True,
        'use_enhanced_encoder': True,
    }
    
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        model, params = load_model(args.checkpoint_dir, config)
    else:
        print("Starting with random weights for RL...")
        model, params = get_empty_model_params(config)
        
    mcts = MCTS(model, params)
    mcts.num_simulations = args.simulations # Helper
    
    all_games_data = []
    for i in range(args.num_games):
        print(f"Playing game {i+1}/{args.num_games}...")
        if args.engine:
            # Note: C++ engine currently uses random/trivial evaluation
            # so this is for testing the pipeline
            game_data = play_self_game_engine(args.engine, args.simulations)
        else:
            game_data = play_self_game(mcts)
        all_games_data.append(game_data)
        
    save_games(all_games_data, args.output_dir)

if __name__ == "__main__":
    main()
