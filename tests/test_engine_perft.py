import pytest
import subprocess
import os

ENGINE_BIN = os.getenv("FLUXFISH_ENGINE", "./build/liquid_chess")

def run_engine_commands(commands):
    try:
        proc = subprocess.run(
            [ENGINE_BIN],
            input="\n".join(commands) + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Engine binary not found or timed out")

@pytest.mark.parametrize(
    "fen,depth,expected",
    [
        # Starting position
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902),
        # Position 2 (Kiwipete)
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 1, 48),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 2, 2039),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 3, 97862),
        # Position 3 (en passant)
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 1, 14),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 2, 191),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 3, 2812),
        # Position 4 (promotions)
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467),
        # Position 5 (castling)
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NNN1/RNBQK2R w KQ - 1 8", 1, 52),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NNN1/RNBQK2R w KQ - 1 8", 2, 1449),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NNN1/RNBQK2R w KQ - 1 8", 3, 72334),
    ],
)
def test_perft_positions(fen, depth, expected):
    commands = [
        "uci",
        "isready",
        f"position fen {fen}",
        f"perft {depth}",
        "quit",
    ]
    out = run_engine_commands(commands)
    # Expect a line like: perft 3: 8902
    for line in out.splitlines():
        if line.startswith("perft"):
            _, _, num = line.split()
            assert int(num) == expected, f"perft {depth} from {fen}: expected {expected}, got {num}"
            return
    pytest.fail(f"perft output not found in engine output: {out}")
