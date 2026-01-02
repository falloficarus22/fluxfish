# Liquid Chess Bot - Complete Roadmap

## Current Status: Basic Training Works ✅
- Model compiles and trains
- Parameters update correctly
- No JAX tracer leaks
- **BUT**: Simplified model, not full LRT yet

---

## Phase 2: Restore Liquid Reasoning Components (1-2 weeks)

### 2.1 Add Back Gating Mechanisms (Carefully)
The key is to implement gates **without causing tracer leaks**:

**Option A: Post-training gates** (Recommended first)
```python
# Train without gates
# Then add gates only during inference with while_loop
def inference_with_gates(self, board_state):
    def cond_fn(carry):
        token, step, stop_prob = carry
        return jnp.logical_and(step < max_steps, stop_prob < 0.95)
    
    def body_fn(carry):
        token, step, _ = carry
        new_token = self.reasoning_step(board_emb, token)
        stop_prob = self.stop_gate(new_token, token)
        return (new_token, step + 1, stop_prob)
    
    return lax.while_loop(cond_fn, body_fn, init_carry)
```

**Option B: Straight-through estimators**
```python
# Gates in training that don't leak tracers
gate_prob = self.discard_gate(old_token, new_token)
# Straight-through: discrete in forward, continuous in backward
keep_discrete = (gate_prob > 0.5).astype(jnp.float32)
keep_smooth = gate_prob + jax.lax.stop_gradient(keep_discrete - gate_prob)
token = keep_smooth * new_token + (1 - keep_smooth) * old_token
```

### 2.2 Enhance Board Encoder
Restore the sophisticated chess-specific features:
- Attack maps (use C++ engine to compute)
- Pin detection
- King safety zones
- Pawn structure features
- Material imbalance encoding

### 2.3 Add Complexity Estimation
Train a separate small network to estimate position complexity:
```python
complexity_net = nn.Dense(1)(board_features)
adaptive_steps = jnp.clip(complexity * max_steps, min_steps, max_steps)
```

---

## Phase 3: Data Pipeline & Training (2-3 weeks)

### 3.1 Gather Training Data
**Sources:**
- Lichess database (millions of games): https://database.lichess.org/
- Download master-level games (2500+ rated)
- Filter for games with time control ≥ 10 minutes
- Extract ~1M positions

**Data format:**
```python
{
    'fen': str,
    'move': str (UCI format),
    'evaluation': float (centipawns),
    'outcome': int (1/0/-1),
    'complexity': float (optional, from engine)
}
```

### 3.2 Create Dataset Pipeline
```python
# Efficient data loading
class ChessDataset:
    def __init__(self, pgn_paths):
        self.positions = self.load_and_cache(pgn_paths)
        
    def load_and_cache(self, paths):
        # Convert PGN -> binary format for fast loading
        # Cache to disk as .npy or .h5
        pass
    
    def augment(self, position):
        # Random flips/rotations
        # Position shuffling
        pass
```

### 3.3 Training Configuration
```yaml
model:
  hidden_dim: 512
  num_heads: 8
  max_steps: 32
  min_steps: 4

training:
  batch_size: 256
  learning_rate: 3e-4
  warmup_steps: 1000
  total_steps: 100000
  
  # Loss weights
  value_weight: 1.0
  policy_weight: 1.0
  step_penalty: 0.01  # Encourage efficiency
  
data:
  num_workers: 4
  prefetch: 8
  cache_size: 100000
```

### 3.4 Training Loop
```bash
# Start training
python train_lrt.py \
    --train data/lichess_master_*.pgn \
    --val data/val.pgn \
    --epochs 10 \
    --batch-size 256 \
    --checkpoint-dir checkpoints/ \
    --use-wandb
```

**Monitor:**
- Value loss (should decrease to ~0.5-1.0)
- Policy accuracy (should reach 30-40%)
- Reasoning steps (should decrease as model learns)
- Validation metrics

---

## Phase 4: C++ Integration (1-2 weeks)

### 4.1 Export Trained Model
```python
# Convert to inference-optimized format
trainer.export_model('models/lrt_chess_v1.pkl')

# Or use ONNX for C++ compatibility
import onnx
onnx_model = convert_to_onnx(trainer.state.params)
```

### 4.2 Bridge C++ Engine with Python Model
File: `cpp/src/bridge/jax_bridge.cpp`
```cpp
class LRTEvaluator {
    py::object jax_model;
    
    float evaluate(const Position& pos) {
        // Convert position to tensor
        auto board_tensor = position_to_tensor(pos);
        
        // Call JAX model
        py::object result = jax_model.attr("evaluate_fast")(board_tensor);
        return result.cast<float>();
    }
};
```

### 4.3 Hybrid Search
Combine LRT with traditional alpha-beta:
```cpp
int search(Position& pos, int depth) {
    // Use LRT for complex positions
    if (is_tactical(pos) && depth > 6) {
        float lrt_eval = lrt_evaluator.evaluate(pos);
        return adjust_search_depth(lrt_eval, depth);
    }
    
    // Use fast NNUE for simple positions
    return nnue_eval(pos);
}
```

---

## Phase 5: Optimization & Tuning (1 week)

### 5.1 Performance Optimization
- Profile with JAX profiler
- Optimize attention (Flash Attention if available)
- Quantize model (FP16 or INT8)
- Batch inference in C++

### 5.2 Search Tuning
- LMR parameters based on LRT complexity
- Time management using LRT confidence
- Multi-PV search for unclear positions

### 5.3 Benchmarking
```bash
# Run perft tests
python tests/test_engine_perft.py

# Speed tests
python tests/test_engine_speed.py

# Play against Stockfish
cutechess-cli -engine cmd=./liquid_chess \
              -engine cmd=stockfish \
              -games 100 -rounds 50
```

---

## Phase 6: Lichess Deployment (1 week)

### 6.1 Create Bot Account
1. Create Lichess account
2. Request BOT account upgrade: https://lichess.org/api#tag/Bot
3. Generate API token

### 6.2 Setup Bot Infrastructure
```bash
# Install lichess-bot
git clone https://github.com/lichess-bot-devs/lichess-bot
cd lichess-bot

# Configure
cp config.yml.default config.yml
# Edit config.yml with your token and engine path
```

### 6.3 Engine Configuration
File: `lichess-bot/engines/liquid_chess.json`
```json
{
    "name": "Liquid Chess v1.0",
    "engine": "./liquid_chess",
    "variants": ["standard"],
    "time_controls": ["bullet", "blitz", "rapid"],
    "uci_options": {
        "Hash": "256",
        "Threads": "4",
        "LRT_Enabled": "true"
    }
}
```

### 6.4 Launch Bot
```bash
# Test locally first
python lichess-bot.py --config config.yml --verbose

# Deploy to server (AWS/GCP/Heroku)
docker build -t liquid-chess-bot .
docker run -d --name lichess-bot liquid-chess-bot
```

### 6.5 Monitoring
- Watch games: https://lichess.org/@/YOUR_BOT
- Track rating progression
- Monitor resource usage
- Collect game data for retraining

---

## Phase 7: Continuous Improvement (Ongoing)

### 7.1 Self-Play Training
```python
# Generate self-play games
python generate_selfplay.py \
    --games 10000 \
    --time-control 1+1 \
    --output data/selfplay/

# Retrain on self-play + master games
python train_lrt.py \
    --train data/selfplay/*.pgn data/lichess_master*.pgn \
    --checkpoint checkpoints/v1.0/ \
    --resume
```

### 7.2 A/B Testing
- Test different LRT architectures
- Tune step penalties
- Experiment with complexity estimation

### 7.3 Competition
- Enter TCEC (Top Chess Engine Championship)
- Participate in CCCC (Computer Chess Championship)
- Compare against other neural engines (Leela, Maia)

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1. Basic Training ✅ | Done | Model trains successfully |
| 2. Restore LRT | 1-2 weeks | Full liquid reasoning |
| 3. Data & Training | 2-3 weeks | Trained model (~2000 Elo) |
| 4. C++ Integration | 1-2 weeks | Fast hybrid search |
| 5. Optimization | 1 week | 100+ NPS, stable |
| 6. Lichess Deploy | 1 week | Live bot |
| 7. Improvement | Ongoing | Rating growth |

**Total to deployment: ~6-9 weeks**

---

## Immediate Next Steps (This Week)

1. **Restore gating mechanisms** (2-3 days)
   - Add back stop gate using `while_loop` for inference
   - Test that it doesn't break training
   
2. **Download training data** (1 day)
   ```bash
   wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
   zstd -d lichess_db_standard_rated_2024-01.pgn.zst
   ```

3. **Setup data pipeline** (2-3 days)
   - Parse PGN files
   - Extract positions + evaluations
   - Cache in efficient format

4. **Start initial training run** (ongoing)
   - Begin with small model (hidden_dim=256)
   - Train overnight
   - Check convergence

---

## Success Metrics

**By deployment:**
- ✅ Model trains without errors
- ✅ Reaches ~2000 Lichess rating (amateur level)
- ✅ Plays legal moves 99.9%+ of time
- ✅ Responds within time limits
- ✅ Handles disconnections gracefully

**Stretch goals:**
- 2200+ rating (expert level)
- Interesting/creative play style
- Efficient reasoning (avg 10-15 steps)
- Competitive with Stockfish at lower depths

---

## Resources Needed

**Compute:**
- GPU for training (RTX 3090 or better, or cloud GPU)
- 4-8 CPU cores for data preprocessing
- 32GB+ RAM

**Storage:**
- ~100GB for training data
- ~50GB for cached datasets
- ~10GB for models/checkpoints

**Tools:**
- Weights & Biases (free tier) for experiment tracking
- GitHub for version control
- Docker for deployment

---

## Questions?

Feel free to ask about:
- Specific implementation details
- Training strategies
- Debugging issues
- Deployment architecture
- Performance optimization