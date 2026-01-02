#!/usr/bin/env python3
import subprocess
import os

ENGINE = "./liquid_chess"

def list_moves(fen):
    cmds = ["uci", "isready", f"position fen {fen}", "go depth 1", "quit"]
    proc = subprocess.run([ENGINE], input="\n".join(cmds) + "\n", capture_output=True, text=True, timeout=30)
    for line in proc.stdout.splitlines():
        if line.startswith("bestmove"):
            return line.split()[1]
    return None

def perft(fen, depth):
    cmds = ["uci", "isready", f"position fen {fen}", f"perft {depth}", "quit"]
    proc = subprocess.run([ENGINE], input="\n".join(cmds) + "\n", capture_output=True, text=True, timeout=30)
    for line in proc.stdout.splitlines():
        if line.startswith("perft"):
            return int(line.split()[2])
    return None

if __name__ == "__main__":
    fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NNN1/RNBQK2R w KQ - 1 8"
    print("FEN:", fen)
    print("perft 1:", perft(fen, 1))
    print("bestmove (depth 1):", list_moves(fen))
