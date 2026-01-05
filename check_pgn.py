import chess.pgn
import hashlib
import os

def check_pgn(pgn_path):
    print(f"🔍 Analyzing {pgn_path}...")
    
    game_count = 0
    unique_games = {} # hash -> game_info
    duplicates = 0
    
    if not os.path.exists(pgn_path):
        print(f"❌ File not found: {pgn_path}")
        return

    file_size = os.path.getsize(pgn_path) / (1024 * 1024)
    print(f"📂 File size: {file_size:.2f} MB")

    with open(pgn_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            game_count += 1
            
            # Create a unique key based on the mainline moves
            # Converting to string of SAN moves is a safe way to identify identical games
            move_list = [move.uci() for move in game.mainline_moves()]
            move_str = ",".join(move_list)
            game_hash = hashlib.md5(move_str.encode()).hexdigest()
            
            if game_hash in unique_games:
                duplicates += 1
            else:
                unique_games[game_hash] = {
                    'white': game.headers.get("White", "?"),
                    'black': game.headers.get("Black", "?"),
                    'result': game.headers.get("Result", "*"),
                    'moves': len(move_list)
                }
            
            if game_count % 500 == 0:
                print(f"   Processed {game_count} games...")

    print(f"\n📊 --- Results ---")
    print(f"✅ Total games found: {game_count}")
    print(f"✨ Unique games: {len(unique_games)}")
    print(f"⚠️  Duplicate games: {duplicates}")
    
    if duplicates > 0:
        print(f"📈 Redundancy: {(duplicates/game_count)*100:.1f}%")

if __name__ == "__main__":
    check_pgn("games.pgn")
