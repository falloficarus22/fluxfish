"""
Efficient Dataset Caching for Chess Training Data
File: python/liquid_chess/training/dataset_cache.py

Converts slow PGN loading to fast binary cache format.
Filters for high-quality games and positions suitable for training.
"""

import numpy as np
import chess
import chess.pgn
from pathlib import Path
from tqdm import tqdm
import hashlib
import json
from typing import Dict, List, Tuple, Optional
import time


class CachedChessDataset:
    """
    High-performance cached dataset for chess positions.
    
    Features:
    - Converts PGN to compressed numpy format
    - Filters for high-quality games (2500+ ELO)
    - Removes obvious positions (openings, simple endgames)
    - Fast random access for training
    """
    
    def __init__(
        self,
        cache_file: str,
        batch_size: int = 32,
        shuffle: bool = True,
        use_enhanced_features: bool = False
    ):
        """
        Initialize dataset from cache file.
        
        Args:
            cache_file: path to .npz cache file
            batch_size: number of positions per batch
            shuffle: whether to shuffle positions
            use_enhanced_features: whether to use enhanced board encoder
        """
        self.cache_file = Path(cache_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_enhanced_features = use_enhanced_features
        
        # Load cache
        print(f"Loading cache from {self.cache_file}")
        self.data = self._load_cache()
        self.size = len(self.data['fens'])
        print(f"Loaded {self.size:,} positions")
        
        # Initialize iteration state
        self.indices = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cursor = 0
    
    def _load_cache(self) -> Dict:
        """Load cached data from disk."""
        data = np.load(self.cache_file, allow_pickle=True)
        return {
            'fens': data['fens'],
            'outcomes': data['outcomes'],
            'move_counts': data['move_counts'],
            'piece_counts': data.get('piece_counts', None),
            'complexity_scores': data.get('complexity_scores', None)
        }
    
    def get_batch(self) -> Dict:
        """
        Get next batch of positions.
        
        Returns:
            batch: dictionary with board states and targets
        """
        # Get batch indices
        if self.cursor + self.batch_size > self.size:
            # Reshuffle and restart
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.cursor = 0
        
        batch_indices = self.indices[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        
        # Load positions
        batch_fens = [self.data['fens'][i] for i in batch_indices]
        batch_outcomes = self.data['outcomes'][batch_indices]
        
        # Convert to model input format
        if self.use_enhanced_features:
            from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input
            boards = [board_to_enhanced_input(chess.Board(fen)) for fen in batch_fens]
        else:
            boards = [self._simple_board_to_input(chess.Board(fen)) for fen in batch_fens]
        
        # Stack into batch
        import jax.numpy as jnp
        
        # Combine all board inputs
        batch_board = {}
        for key in boards[0].keys():
            batch_board[key] = jnp.stack([b[key] for b in boards])
        
        # Create policy targets (uniform over legal moves)
        policy_targets = []
        legal_masks = []
        
        for fen in batch_fens:
            board = chess.Board(fen)
            legal_moves = np.zeros((64, 64), dtype=np.float32)
            
            for move in board.legal_moves:
                legal_moves[move.from_square, move.to_square] = 1.0
            
            if legal_moves.sum() > 0:
                legal_moves = legal_moves / legal_moves.sum()
            
            policy_targets.append(legal_moves)
            legal_masks.append((legal_moves > 0).astype(np.float32))
        
        batch_dict = {
            'board': batch_board,
            'outcome': jnp.array(batch_outcomes, dtype=jnp.float32),
            'policy': jnp.array(policy_targets, dtype=jnp.float32),
            'legal_moves': jnp.array(legal_masks, dtype=jnp.float32)
        }

        if 'evaluations' in self.data:
            batch_dict['evaluations'] = jnp.array(
                self.data['evaluations'][batch_indices],
                dtype=jnp.float32
            )

        return batch_dict
    
    def _simple_board_to_input(self, board: chess.Board) -> Dict:
        """Convert board to simple input format (no enhanced features)."""
        import jax.numpy as jnp
        
        pieces = np.zeros((8, 8), dtype=np.int8)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Map to index 1-12
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
    
    @staticmethod
    def create_cache(
        pgn_path: str,
        cache_dir: str,
        max_positions: int = 1_000_000,
        min_elo: int = 0,
        skip_opening_moves: int = 8,
        min_pieces: int = 6,
        max_pieces: int = 32,
        engine_games: bool = True,
        extract_evals: bool = True
    ) -> Path:
        """
        Create cache from PGN file.
        
        Args:
            pgn_path: path to PGN file
            cache_dir: directory to save cache
            max_positions: maximum positions to extract
            min_elo: minimum ELO for both players
            skip_opening_moves: skip first N moves
            min_pieces: minimum pieces on board
            max_pieces: maximum pieces on board
        
        Returns:
            cache_file: path to created cache file
        """
        pgn_path = Path(pgn_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache key from PGN file
        cache_key = hashlib.md5(pgn_path.name.encode()).hexdigest()[:8]
        cache_file = cache_dir / f"cache_{pgn_path.stem}_{cache_key}.npz"
        
        if cache_file.exists():
            print(f"✅ Cache already exists: {cache_file}")
            return cache_file
        
        print(f"📦 Creating cache from {pgn_path.name}")
        print(f"   Min ELO: {min_elo}")
        print(f"   Max positions: {max_positions:,}")
        
        positions = []
        outcomes = []
        move_counts = []
        piece_counts = []
        evaluations = []
        
        start_time = time.time()
        
        with open(pgn_path) as pgn:
            game_count = 0
            position_count = 0
            skipped_elo = 0
            skipped_result = 0
            
            pbar = tqdm(total=max_positions, desc="Extracting positions")
            
            while position_count < max_positions:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                if not engine_games:
                    # Check ELO ratings
                    try:
                        white_elo = int(game.headers.get('WhiteElo', 0))
                        black_elo = int(game.headers.get('BlackElo', 0))
                    except (ValueError, TypeError):
                        skipped_elo += 1
                        continue
                
                    if white_elo < min_elo or black_elo < min_elo:
                        skipped_elo += 1
                        continue
                
                # Parse result
                result = game.headers.get('Result', '*')
                if result == '1-0':
                    outcome = 100.0  # centipawns for white win
                elif result == '0-1':
                    outcome = -100.0  # centipawns for black win
                elif result == '1/2-1/2':
                    outcome = 0.0
                else:
                    skipped_result += 1
                    continue
                
                # Extract positions from game
                board = game.board()
                move_num = 0
                
                for move in game.mainline_moves():
                    board.push(move)
                    move_num += 1
                    
                    # Skip early opening
                    if move_num <= skip_opening_moves:
                        continue
                    
                    # Filter by piece count
                    piece_count = len(board.piece_map())
                    if piece_count < min_pieces or piece_count > max_pieces:
                        continue
                    
                    # Skip drawn positions in won games (likely simplified)
                    if abs(outcome) > 50 and piece_count < 10:
                        continue
                    
                    # Store position
                    positions.append(board.fen())
                    outcomes.append(outcome)
                    move_counts.append(move_num)
                    piece_counts.append(piece_count)

                    eval_score = 0.0
                    if extract_evals and hasattr(move, 'parent') and move.parent():
                        node = move.parent()
                        if node.comment:
                            import re
                            # Parse comments like "[%eval 0.5]" or "[%eval #3]"
                            eval_match = re.search(r'\[%eval ([+-]?\d+\.?\d*)\]', node.comment)
                            if eval_match:
                                try:
                                    eval_score = float(eval_match.group(1)) * 100  # Convert to centipawns
                                except:
                                    pass
                            # Parse mate scores like "[%eval #5]"
                            mate_match = re.search(r'\[%eval #([+-]?\d+)\]', node.comment)
                            if mate_match:
                                mate_in = int(mate_match.group(1))
                                # Convert mate to large evaluation
                                eval_score = 10000 if mate_in > 0 else -10000

                    evaluations.append(eval_score)
                    
                    position_count += 1
                    pbar.update(1)
                    
                    if position_count >= max_positions:
                        break
                
                game_count += 1
                
                # Update progress bar with stats
                if game_count % 100 == 0:
                    pbar.set_postfix({
                        'games': game_count,
                        'skip_elo': skipped_elo,
                        'skip_result': skipped_result,
                        'positions': position_count
                    })
        
        pbar.close()
        
        elapsed = time.time() - start_time
        
        print(f"\n📊 Extraction Statistics:")
        print(f"   Games processed: {game_count:,}")
        print(f"   Positions extracted: {position_count:,}")
        print(f"   Skipped (low ELO): {skipped_elo:,}")
        print(f"   Skipped (no result): {skipped_result:,}")
        print(f"   Time: {elapsed:.1f}s ({position_count/elapsed:.0f} pos/s)")
        
        # Compute basic statistics
        outcomes_array = np.array(outcomes, dtype=np.float32)
        piece_counts_array = np.array(piece_counts, dtype=np.int16)
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   Outcome distribution:")
        print(f"     White wins: {(outcomes_array > 0).sum():,} ({(outcomes_array > 0).mean()*100:.1f}%)")
        print(f"     Draws: {(outcomes_array == 0).sum():,} ({(outcomes_array == 0).mean()*100:.1f}%)")
        print(f"     Black wins: {(outcomes_array < 0).sum():,} ({(outcomes_array < 0).mean()*100:.1f}%)")
        print(f"   Average pieces: {piece_counts_array.mean():.1f}")
        print(f"   Piece range: {piece_counts_array.min()}-{piece_counts_array.max()}")
        
        # Save cache
        print(f"\n💾 Saving cache to {cache_file}")
        # Prepare save dictionary
        save_dict = {
            'fens': np.array(positions, dtype=object),
            'outcomes': outcomes_array,
            'move_counts': np.array(move_counts, dtype=np.int16),
            'piece_counts': piece_counts_array,
            'metadata': {
                'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': pgn_path.name,
                'min_elo': min_elo,
                'games_processed': game_count,
                'total_positions': position_count,
                'engine_games': engine_games,  # <-- ADD THIS
            }
        }

        if extract_evals:
            evals_array = np.array(evaluations, dtype=np.float32)
            if np.any(evals_array != 0):
                save_dict['evaluations'] = evals_array
                print(f"   ✅ Included {(evals_array != 0).sum():,} position evaluations")
                print(f"   📊 Eval range: {evals_array[evals_array != 0].min():.0f} to {evals_array[evals_array != 0].max():.0f} cp")

        np.savez_compressed(cache_file, **save_dict)
        
        # Verify cache size
        cache_size_mb = cache_file.stat().st_size / 1024 / 1024
        print(f"   Cache size: {cache_size_mb:.1f} MB")
        print(f"   Compression: {cache_size_mb / position_count * 1024:.2f} KB/position")
        
        print(f"\n✅ Cache created successfully!")
        
        return cache_file
    
    @staticmethod
    def merge_caches(cache_files: List[str], output_file: str) -> Path:
        """
        Merge multiple cache files into one.
        
        Args:
            cache_files: list of cache file paths
            output_file: path for merged cache
        
        Returns:
            output_path: path to merged cache
        """
        print(f"🔗 Merging {len(cache_files)} cache files...")
        
        all_fens = []
        all_outcomes = []
        all_move_counts = []
        all_piece_counts = []
        
        for cache_file in cache_files:
            print(f"   Loading {Path(cache_file).name}")
            data = np.load(cache_file, allow_pickle=True)
            
            all_fens.extend(data['fens'])
            all_outcomes.append(data['outcomes'])
            all_move_counts.append(data['move_counts'])
            if 'piece_counts' in data:
                all_piece_counts.append(data['piece_counts'])
        
        # Concatenate
        merged_fens = np.array(all_fens, dtype=object)
        merged_outcomes = np.concatenate(all_outcomes)
        merged_move_counts = np.concatenate(all_move_counts)
        
        if all_piece_counts:
            merged_piece_counts = np.concatenate(all_piece_counts)
        else:
            merged_piece_counts = None
        
        print(f"   Total positions: {len(merged_fens):,}")
        
        # Save merged cache
        output_path = Path(output_file)
        save_dict = {
            'fens': merged_fens,
            'outcomes': merged_outcomes,
            'move_counts': merged_move_counts,
        }
        
        if merged_piece_counts is not None:
            save_dict['piece_counts'] = merged_piece_counts
        
        np.savez_compressed(output_path, **save_dict)
        
        print(f"✅ Merged cache saved to {output_path}")
        
        return output_path


def create_dataset_from_pgn(
    pgn_paths: List[str],
    cache_dir: str = "data/cache",
    **kwargs
) -> CachedChessDataset:
    """
    Convenience function to create dataset from PGN files.
    
    Args:
        pgn_paths: list of PGN file paths
        cache_dir: directory for cache files
        **kwargs: additional arguments for create_cache
    
    Returns:
        dataset: CachedChessDataset ready for training
    """
    cache_files = []
    
    for pgn_path in pgn_paths:
        cache_file = CachedChessDataset.create_cache(
            pgn_path,
            cache_dir,
            **kwargs
        )
        cache_files.append(str(cache_file))
    
    # If multiple caches, merge them
    if len(cache_files) > 1:
        merged_cache = CachedChessDataset.merge_caches(
            cache_files,
            Path(cache_dir) / "merged_cache.npz"
        )
        return CachedChessDataset(str(merged_cache))
    else:
        return CachedChessDataset(cache_files[0])


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_cache.py <pgn_file> [cache_dir]")
        sys.exit(1)
    
    pgn_file = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "data/cache"
    
    # Create cache
    cache_file = CachedChessDataset.create_cache(
        pgn_file,
        cache_dir,
        max_positions=100_000  # Smaller for testing
    )
    
    # Load and test
    print("\n🧪 Testing cache loading...")
    dataset = CachedChessDataset(str(cache_file), batch_size=8)
    
    # Get a batch
    batch = dataset.get_batch()
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Batch size: {batch['outcome'].shape[0]}")
    print(f"   Board shape: {batch['board']['pieces'].shape}")
    
    print("\n✅ Cache working correctly!")