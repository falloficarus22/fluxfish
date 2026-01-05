import chess.pgn
import hashlib
import os

def deduplicate_pgn(input_path, output_path):
    print(f"🧹 Deduplicating {input_path} -> {output_path}...")
    
    unique_hashes = set()
    total = 0
    saved = 0
    
    with open(input_path) as pgn_in, open(output_path, "w") as pgn_out:
        while True:
            game = chess.pgn.read_game(pgn_in)
            if game is None:
                break
            
            total += 1
            # Hash the moves to detect identical games
            move_str = ",".join([move.uci() for move in game.mainline_moves()])
            game_hash = hashlib.md5(move_str.encode()).hexdigest()
            
            if game_hash not in unique_hashes:
                unique_hashes.add(game_hash)
                # Export the game to the new file
                print(game, file=pgn_out, end="\n\n")
                saved += 1
            
            if total % 500 == 0:
                print(f"   Processed {total} games, found {saved} unique...")

    print(f"\n✨ --- Cleanup Complete ---")
    print(f"✅ Total games scanned: {total}")
    print(f"✨ Unique games saved:  {saved}")
    print(f"🗑️  Duplicates removed: {total - saved}")

if __name__ == "__main__":
    deduplicate_pgn("games.pgn", "games_unique.pgn")
