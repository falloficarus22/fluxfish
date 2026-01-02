import os
import subprocess
import time
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _engine_path() -> Path:
    env = os.environ.get("FLUXFISH_ENGINE")
    if env:
        return Path(env)
    return _repo_root() / "liquid_chess"


def _start_engine() -> subprocess.Popen[str]:
    engine = _engine_path()
    if not engine.exists():
        pytest.skip(f"engine binary not found at {engine} (set FLUXFISH_ENGINE to override)")
    return subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _send(proc: subprocess.Popen[str], cmd: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def _read_until(proc: subprocess.Popen[str], predicate, timeout_s: float = 5.0) -> list[str]:
    assert proc.stdout is not None
    lines: list[str] = []
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line:
            lines.append(line)
        if predicate(line):
            return lines
    raise TimeoutError(f"timeout waiting for engine output; got: {lines[-10:]}")


def _uci_handshake(proc: subprocess.Popen[str]) -> None:
    _send(proc, "uci")
    _read_until(proc, lambda l: l == "uciok", timeout_s=5.0)
    _send(proc, "isready")
    _read_until(proc, lambda l: l == "readyok", timeout_s=5.0)


@pytest.mark.slow
def test_engine_depth1_speed_smoke():
    chess = pytest.importorskip("chess")

    proc = _start_engine()
    try:
        _uci_handshake(proc)

        # A small set of varied positions (middlegame-ish)
        fens = [
            chess.STARTING_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 2",
            "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 2 7",
            "4rrk1/ppp2ppp/2n5/3q4/3P4/2P2N2/PP3PPP/R2QR1K1 w - - 0 16",
        ]

        t0 = time.time()
        for fen in fens:
            _send(proc, f"position fen {fen}")
            _send(proc, "go depth 1")
            _read_until(proc, lambda l: l.startswith("bestmove "), timeout_s=10.0)

        elapsed = time.time() - t0
        # Loose bound: just ensure it doesn't hang / isn't absurdly slow.
        assert elapsed < 10.0

        print({"depth1_positions": len(fens), "elapsed_s": elapsed, "pos_per_s": len(fens) / max(elapsed, 1e-9)})
    finally:
        try:
            _send(proc, "quit")
        except Exception:
            pass
        proc.kill()
