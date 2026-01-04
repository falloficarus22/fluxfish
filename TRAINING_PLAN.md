# FluxFish Training Plan

This document outlines the systematic training phases for the **FluxFish** chess engine, leveraging the Liquid Reasoning Transformer (LRT) architecture.

---

## 🎯 Objectives
- Reach **2500+ Lichess Blitz** Elo.
- Achieve human-like tactical intuition with engine-level precision.
- Optimize the LRT gating mechanism for efficient inference (adaptive reasoning).

---

## 💻 Infrastructure Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3060 (12GB) | NVIDIA RTX 4090 or A100 |
| **RAM** | 32 GB | 64 GB+ |
| **Storage** | 200 GB SSD | 1 TB NVMe |
| **CPU** | 8 Cores | 16+ Cores (for data processing) |

---

## 📈 Phase 1: Engine Distillation (Immediate)
**Current Status:** 720 Stockfish Self-Play games generated ✅

**Goal:** Train the initial model to mimic Stockfish 16 evaluations and move preferences using the existing `games.pgn`.

### 1.1 Data Processing
- **Input:** `games.pgn` (720 games, ~100k positions).
- **Tool:** Use `dataset_cache.py` to extract positions and evaluations into a binary format.
- **Filtering:** 
    - Keep positions with Stockfish evaluations.
    - Filter out simple draws to focus on decisive tactical play.

### 1.2 Training Methodology
- **Targets:**
    - **Policy:** The move suggested/played by Stockfish.
    - **Value:** Stockfish's centipawn evaluation (normalized).
- **Objective:** Establish the "Intuition" base for the Liquid Reasoning Transformer.

---

## 🔝 Phase 2: Supervised Learning (SL) - Scaling
**Duration:** 1-2 weeks
**Goal:** Expand the knowledge base using millions of Elite human games.

### 2.1 Data Generation
- Take positions from Phase 1 and re-evaluate them using **Stockfish 16.1**.
- **Settings:** `Depth 20` or `100ms per position`.
- **Output:** Centipawn evaluation + Best Move.

### 2.2 Training Methodology
- **Targets:**
    - **Value:** Sigmoid-normalized centipawn score: $v = \tanh(cp / 100.0)$.
    - **Policy:** Stockfish's suggested best move.
- **LRT Specifics:** 
    - Start training the **Stop Gate** to predict how many reasoning steps are needed for a given complexity.

---

## 🌊 Phase 3: Liquid Reasoning & RL
**Duration:** Ongoing
**Goal:** Optimize reasoning steps and surpass teacher limits via self-play.

### 3.1 Adaptive Gating
- Train the **Stop Gate** to intelligently decide when to "stop thinking."
- **Loss:** $L = L_{policy} + L_{value} + 0.01 \cdot \text{avg\_steps}$.

### 3.2 MCTS Self-Play
- Use the high-performance C++ backend for fast game generation.
- Reinforce successful tactical sequences found during home-grown play.

---

## 🛠️ Execution Roadmap (Action Items)

### 1. Prepare existing Stockfish Data
Run the caching script on your 720 games:
```bash
python python/liquid_chess/training/dataset_cache.py games.pgn data/cache/
```

### 2. Verify Cache
Ensure `data/cache/` contains the `.npz` files:
```bash
ls -lh data/cache/
```

### 3. Launch Initial Training
```bash
python train_lrt.py `
    --train games.pgn `
    --epochs 50 `
    --batch-size 256 `
    --steps-per-epoch 1000 `
    --use-wandb
```
*(Note: If training from raw PGN, the script will handle caching automatically.)*

---

## 📊 Success Metrics
- **Validation Accuracy (Policy):** > 45% on Elite games.
- **Value MSE:** < 0.10.
- **Reasoning Efficiency:** Avg steps < 12 for non-tactical positions.
- **Engine Strength:** Win rate > 50% against Stockfish Level 5 (at 100ms/move).
