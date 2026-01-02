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


def _bestmove(proc: subprocess.Popen[str], position_cmd: str, depth: int = 1) -> str:
    _send(proc, position_cmd)
    _send(proc, f"go depth {depth}")
    lines = _read_until(proc, lambda l: l.startswith("bestmove "), timeout_s=10.0)
    best = [l for l in lines if l.startswith("bestmove ")][-1]
    return best.split()[1]


@pytest.mark.parametrize(
    "position_cmd",
    [
        "position startpos",
        "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "position fen 8/8/8/3pP3/8/8/8/8 w - d6 0 1",
        "position fen 4k3/8/8/8/8/8/4R3/4K3 b - - 0 1",
    ],
)
def test_engine_bestmove_is_legal(position_cmd: str):
    chess = pytest.importorskip("chess")

    proc = _start_engine()
    try:
        _uci_handshake(proc)

        move_uci = _bestmove(proc, position_cmd, depth=1)

        if position_cmd == "position startpos":
            board = chess.Board()
        else:
            fen = position_cmd.split("position fen ", 1)[1]
            board = chess.Board(fen)

        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves
    finally:
        try:
            _send(proc, "quit")
        except Exception:
            pass
        proc.kill()
