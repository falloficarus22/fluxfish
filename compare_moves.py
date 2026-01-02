#!/usr/bin/env python3
import subprocess
import chess

ENGINE = "./liquid_chess"

def our_legal_moves(fen):
    cmds = ["uci", "isready", f"position fen {fen}", "dumplegal", "quit"]
    proc = subprocess.run([ENGINE], input="\n".join(cmds) + "\n", capture_output=True, text=True, timeout=30)
    moves = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        # UCI move strings are 4 or 5 chars like e2e4 or a7a8q
        if 4 <= len(line) <= 5 and line[0] in "abcdefgh" and line[2] in "abcdefgh":
            moves.append(line)
    return set(moves)

def python_chess_legal_moves(fen):
    board = chess.Board(fen)
    return {move.uci() for move in board.legal_moves}

if __name__ == "__main__":
    fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NNN1/RNBQK2R w KQ - 1 8"
    ours = our_legal_moves(fen)
    pc = python_chess_legal_moves(fen)
    print("Our moves:", len(ours))
    print("python-chess moves:", len(pc))
    print("Extra moves (ours - python-chess):")
    for m in sorted(ours - pc):
        print(" ", m)
    print("Missing moves (python-chess - ours):")
    for m in sorted(pc - ours):
        print(" ", m)
