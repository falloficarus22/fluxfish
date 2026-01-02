#!/usr/bin/env python3
import chess

def perft(board, depth):
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes

if __name__ == "__main__":
    fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NNN1/RNBQK2R w KQ - 1 8"
    board = chess.Board(fen)
    for d in range(1, 4):
        print(f"perft {d}:", perft(board.copy(), d))
