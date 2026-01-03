#!/usr/bin/env python3
"""
Generate Stockfish self-play games for training
No cutechess-cli required - pure Python!
"""

import chess
import chess.engine
import chess.pgn
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
import random
import sys

class StockfishGameGenerator:
    """Generate high-quality self-play games using Stockfish."""
    
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        time_limit: float = 1.0,  # seconds per move
        depth_limit: int = None,  # or use depth instead
        threads: int = 4,
        hash_mb: int = 512,
    ):
        """
        Initialize generator.
        
        Args:
            stockfish_path: path to stockfish binary
            time_limit: seconds per move (None to use depth)
            depth_limit: search depth (None to use time)
            threads: CPU threads
            hash_mb: hash table size in MB
        """
        self.stockfish_path = stockfish_path
        self.time_limit = time_limit
        self.depth_limit = depth_limit
        self.threads = threads
        self.hash_mb = hash_mb
        
        # Opening book (simple starting positions for variety)
        self.opening_book = [
            # Standard openings
            "e2e4",  # King's pawn
            "d2d4",  # Queen's pawn
            "c2c4",  # English
            "g1f3",  # Reti
            # Italian
            "e2e4 e7e5 g1f3 b8c6 f1c4",
            # Spanish
            "e2e4 e7e5 g1f3 b8c6 f1b5",
            # Sicilian
            "e2e4 c7c5",
            # French
            "e2e4 e7e6",
            # Caro-Kann
            "e2e4 c7c6",
            # Queen's Gambit
            "d2d4 d7d5 c2c4",
            # King's Indian
            "d2d4 g8f6 c2c4 g7g6",
            # Nimzo-Indian
            "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
        ]
    
    def generate_games(
        self,
        num_games: int,
        output_file: str,
        include_eval: bool = True,
        max_moves: int = 200,
    ):
        """
        Generate self-play games.
        
        Args:
            num_games: number of games to generate
            output_file: output PGN file path
            include_eval: include evaluation in comments
            max_moves: maximum moves per game (prevent infinite games)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🤖 Generating {num_games} Stockfish self-play games")
        print(f"   Output: {output_file}")
        print(f"   Time/move: {self.time_limit}s")
        print(f"   Depth: {self.depth_limit if self.depth_limit else 'N/A'}")
        print(f"   Threads: {self.threads}")
        print(f"   Hash: {self.hash_mb} MB")
        print()
        
        # Test stockfish
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as test_engine:
                print(f"✅ Stockfish found: {test_engine.id['name']}")
        except Exception as e:
            print(f"❌ Error starting Stockfish: {e}")
            print(f"   Path: {self.stockfish_path}")
            print()
            print("Install Stockfish:")
            print("  macOS:  brew install stockfish")
            print("  Ubuntu: sudo apt-get install stockfish")
            return
        
        print()
        
        # Estimate time
        est_seconds = num_games * max_moves * self.time_limit / 30  # ~30 moves per game avg
        est_hours = est_seconds / 3600
        print(f"⏱️  Estimated time: {est_hours:.1f} hours")
        print()
        
        # Generate games
        start_time = time.time()
        
        with open(output_file, 'w') as pgn_out:
            for game_num in tqdm(range(num_games), desc="Generating games"):
                try:
                    game = self._generate_single_game(
                        game_num + 1,
                        include_eval,
                        max_moves
                    )
                    
                    if game:
                        print(game, file=pgn_out)
                        pgn_out.flush()
                    
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user")
                    break
                except Exception as e:
                    print(f"\n⚠️  Error in game {game_num + 1}: {e}")
                    continue
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Generation complete!")
        print(f"   Time: {elapsed/60:.1f} minutes")
        print(f"   Games generated: {game_num + 1}")
        print(f"   Output: {output_file}")
    
    def _generate_single_game(
        self,
        game_number: int,
        include_eval: bool,
        max_moves: int,
    ) -> chess.pgn.Game:
        """Generate a single game."""
        
        # Create new game
        game = chess.pgn.Game()
        game.headers["Event"] = "Stockfish Self-Play"
        game.headers["Site"] = "Training Data Generation"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_number)
        game.headers["White"] = "Stockfish"
        game.headers["Black"] = "Stockfish"
        
        # Start from opening position
        opening = random.choice(self.opening_book)
        board = chess.Board()
        node = game
        
        # Play opening moves
        for move_uci in opening.split():
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                node = node.add_variation(move)
            except:
                break
        
        # Start engine
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            # Configure engine
            engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb,
            })
            
            # Play game
            move_count = 0
            
            while not board.is_game_over() and move_count < max_moves:
                # Determine time limit
                if self.depth_limit:
                    limit = chess.engine.Limit(depth=self.depth_limit)
                else:
                    limit = chess.engine.Limit(time=self.time_limit)
                
                # Get best move with evaluation
                result = engine.play(
                    board,
                    limit,
                    info=chess.engine.INFO_ALL
                )
                
                move = result.move
                board.push(move)
                node = node.add_variation(move)
                
                # Add evaluation to comment if requested
                if include_eval and hasattr(result, 'info') and 'score' in result.info:
                    score = result.info['score']
                    
                    # Format evaluation
                    if score.is_mate():
                        mate_in = score.relative.mate()
                        node.comment = f"[%eval #{mate_in}]"
                    else:
                        cp = score.relative.score()
                        eval_pawns = cp / 100.0
                        node.comment = f"[%eval {eval_pawns:.2f}]"
                
                move_count += 1
        
        # Set result
        result = board.result()
        game.headers["Result"] = result
        
        # Add termination reason
        if board.is_checkmate():
            game.headers["Termination"] = "checkmate"
        elif board.is_stalemate():
            game.headers["Termination"] = "stalemate"
        elif board.is_insufficient_material():
            game.headers["Termination"] = "insufficient material"
        elif board.is_fifty_moves():
            game.headers["Termination"] = "50-move rule"
        elif board.is_repetition():
            game.headers["Termination"] = "threefold repetition"
        elif move_count >= max_moves:
            game.headers["Termination"] = "move limit"
        
        return game


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Stockfish self-play games")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of games to generate (default: 100)")
    parser.add_argument("--output", type=str, 
                       default="data/engine_games/stockfish_selfplay/games.pgn",
                       help="Output PGN file")
    parser.add_argument("--stockfish", type=str, default="stockfish",
                       help="Path to Stockfish binary")
    parser.add_argument("--time", type=float, default=1.0,
                       help="Time per move in seconds (default: 1.0)")
    parser.add_argument("--depth", type=int, default=None,
                       help="Search depth (overrides time)")
    parser.add_argument("--threads", type=int, default=4,
                       help="CPU threads (default: 4)")
    parser.add_argument("--hash", type=int, default=512,
                       help="Hash size in MB (default: 512)")
    parser.add_argument("--no-eval", action="store_true",
                       help="Don't include evaluations in comments")
    parser.add_argument("--max-moves", type=int, default=200,
                       help="Max moves per game (default: 200)")
    
    args = parser.parse_args()
    
    generator = StockfishGameGenerator(
        stockfish_path=args.stockfish,
        time_limit=args.time,
        depth_limit=args.depth,
        threads=args.threads,
        hash_mb=args.hash,
    )
    
    generator.generate_games(
        num_games=args.games,
        output_file=args.output,
        include_eval=not args.no_eval,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()