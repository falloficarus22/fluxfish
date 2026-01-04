import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
import chex
from functools import partial
from .enhanced_encoder import EnhancedChessBoardEncoder
from .gates import AdaptiveGates

class ChessBoardEncoder(nn.Module):
    """Ultra-efficient chess board encoder"""
    features: int = 512
    
    @nn.compact
    def __call__(self, board_state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Input shapes:
        # pieces: [8, 8] int8 (0-12)
        # turn: scalar bool
        # castling: [4] bool
        # ep_square: scalar int8 (-1 or 0-63)
        
        # 1. Piece embeddings (13 piece types * 64 squares)
        piece_emb = nn.Embed(13, self.features // 8)(board_state['pieces'].flatten())
        piece_emb = piece_emb.reshape(64, -1)  # [64, features//8]
        
        # 2. Square features (chess-specific)
        # Create rank and file features
        ranks = jnp.arange(8).repeat(8).reshape(64, 1) / 7.0
        files = jnp.tile(jnp.arange(8), 8).reshape(64, 1) / 7.0
        
        # Center control
        center_dist = jnp.abs(ranks - 3.5) + jnp.abs(files - 3.5)
        center_dist = center_dist / 7.0
        
        # King safety zones
        back_rank = jnp.logical_or(ranks == 0, ranks == 7).astype(jnp.float32)
        
        square_features = jnp.concatenate([
            ranks, files, center_dist, back_rank
        ], axis=-1)
        
        # 3. Positional embeddings (learned)
        pos_emb = nn.Dense(self.features // 8)(square_features)
        
        # 4. Dynamic features (attacks, pins, etc.)
        # Attack maps
        white_attacks = board_state.get('white_attacks', jnp.zeros((64,), dtype=jnp.bool_))
        black_attacks = board_state.get('black_attacks', jnp.zeros((64,), dtype=jnp.bool_))
        
        attack_features = jnp.stack([
            white_attacks.astype(jnp.float32),
            black_attacks.astype(jnp.float32)
        ], axis=-1)
        
        attack_emb = nn.Dense(self.features // 8)(attack_features)
        
        # 5. Metadata (turn, castling, ep)
        turn_feat = board_state['turn'].astype(jnp.float32).reshape(1, 1)
        castling_feat = board_state['castling'].astype(jnp.float32).reshape(1, 4)
        ep_feat = (board_state['ep_square'] >= 0).astype(jnp.float32).reshape(1, 1)
        
        meta = jnp.concatenate([turn_feat, castling_feat, ep_feat], axis=-1)
        meta_emb = nn.Dense(self.features // 8)(meta)
        meta_emb = jnp.tile(meta_emb, (64, 1))  # Broadcast to all squares
        
        # 6. Combine all features
        combined = jnp.concatenate([
            piece_emb,      # [64, f//8]
            pos_emb,        # [64, f//8]
            attack_emb,     # [64, f//8]
            meta_emb        # [64, f//8]
        ], axis=-1)  # [64, features]
        
        # Final projection
        encoded = nn.Dense(self.features)(combined)
        encoded = nn.LayerNorm()(encoded)
        
        return encoded

class UltraFastLRT(nn.Module):
    """Complete Liquid Reasoning Transformer for chess - Simplified for training"""
    config: Dict[str, Any]

    def setup(self):
        hidden_dim = self.config['hidden_dim']
        self.gates = AdaptiveGates(hidden_dim)
        
        # Board Encoders
        self.enhanced_encoder = EnhancedChessBoardEncoder(hidden_dim)
        self.simple_pieces_emb = nn.Embed(13, hidden_dim // 4)
        self.simple_pos_enc = nn.Dense(hidden_dim // 4)
        self.simple_comb_proj = nn.Dense(hidden_dim)
        self.simple_ln = nn.LayerNorm()
        
        # Transformer components
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config['num_heads'],
            qkv_features=hidden_dim,
            dropout_rate=self.config.get('dropout_rate', 0.0),
        )
        self.ln1 = nn.LayerNorm()
        self.ff1 = nn.Dense(hidden_dim * 4)
        self.ff2 = nn.Dense(hidden_dim)
        self.ln2 = nn.LayerNorm()

        # Output Heads
        self.value_head = nn.Dense(1)
        self.policy_head = nn.Dense(64 * 64)

        # Reasoning token
        self.init_token_param = self.param('init_token',
                                         random.normal,
                                         (1, hidden_dim))
    
    def __call__(self, 
                 board_state: Dict[str, jnp.ndarray],
                 max_steps: int = 50,
                 deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        
        hidden_dim = self.config['hidden_dim']
        use_enhanced_encoder = self.config.get('use_enhanced_encoder', False)

        if use_enhanced_encoder:
            board_emb = self.enhanced_encoder(board_state)
        else:
            pieces_flat = board_state['pieces'].flatten()
            pieces_emb = self.simple_pieces_emb(pieces_flat)
            pos_enc = self.simple_pos_enc(pieces_emb)
            board_emb = self.simple_comb_proj(jnp.concatenate([pieces_emb, pos_enc], axis=-1))
            board_emb = self.simple_ln(board_emb)
        
        # Initialize reasoning token
        init_token = self.init_token_param
        
        complexity = jnp.reshape(self.gates.estimate_complexity(board_emb), ())
        adaptive_max_steps = jnp.reshape(self.gates.compute_adaptive_steps(
            complexity, 
            min_steps=4, 
            max_steps=max_steps
        ), ())
        
        # Warmup call to initialize all parameters outside the symbolic while_loop.
        # This prevents JAX UnexpectedTracerError during init/jit.
        self._warmup(board_emb, init_token, deterministic)
        
        # Initial loop state: token, total_keep_prob, should_continue, actual_steps
        init_state = (init_token, 0.0, True, 0)
        
        def scan_fn(state, step_idx):
            token, current_keep_prob, should_continue, actual_steps = state
            
            # We continue if we haven't stopped AND we are within the adaptive budget
            do_step = should_continue & (step_idx < adaptive_max_steps)
            
            def true_fn(args):
                t, ckp, _, stps = args
                
                # 1. Transformer step
                new_token = self.transformer_step(board_emb, t, deterministic)
                
                # 2. Apply discard gate
                gate_rng = self.make_rng('dropout') if not deterministic else None
                updated_token, keep_prob = self.gates.apply_discard_gate(
                    t, new_token, deterministic, rng=gate_rng
                )
                
                # 3. Check stop gate
                stop_prob = self.gates.should_stop(updated_token, step_idx)
                
                keep_prob_scalar = jnp.reshape(keep_prob, ())
                next_should_continue = jnp.reshape(stop_prob < 0.95, ())
                
                return updated_token, ckp + keep_prob_scalar, next_should_continue, stps + 1

            def false_fn(args):
                return args # Just pass through

            new_state = jax.lax.cond(do_step, true_fn, false_fn, (token, current_keep_prob, should_continue, actual_steps))
            return new_state, None

        # Execute the scan for exactly max_steps (maximal possible)
        (current_token, total_keep_prob, _, final_step), _ = jax.lax.scan(
            scan_fn, init_state, jnp.arange(max_steps)
        )
        
        # Output heads
        value = self.value_head(current_token).squeeze()
        policy_logits = self.policy_head(current_token).squeeze()
        
        # Format outputs (Value is normalized -1 to 1, Policy is probability dist)
        value_normalized = jnp.tanh(value)
        policy = nn.softmax(policy_logits).reshape(64, 64)
        
        # Collect all statistics
        stats = {
            'steps_taken': final_step,
            'avg_keep_prob': total_keep_prob / jnp.maximum(final_step, 1.0),
            'final_stop_prob': jnp.reshape(self.gates.should_stop(current_token, final_step), ()),
        }
        
        return {
            'value': value_normalized,
            'policy': policy,
            'stats': stats,
            'final_token': current_token
        }

    def _warmup(self, board_emb, token, deterministic):
        """Warm up submodules to ensure they are initialized before while_loop."""
        _ = self.transformer_step(board_emb, token, deterministic)
        gate_rng = self.make_rng('dropout') if not deterministic else None
        _ = self.gates.apply_discard_gate(token, token, deterministic, rng=gate_rng)
        _ = self.gates.should_stop(token, 0)


    def transformer_step(self, board_emb, token, deterministic):
        """Single transformer pass with board-reasoning attention."""
        # Concatenate board and token: [64, d] + [1, d] -> [65, d]
        combined_tokens = jnp.concatenate([board_emb, token], axis=0)
        
        # Apply attention and FFN
        attn_out = self.attn(combined_tokens, combined_tokens, deterministic=deterministic)
        x = self.ln1(combined_tokens + attn_out)
        
        ff_out = self.ff1(x)
        ff_out = nn.relu(ff_out)
        ff_out = self.ff2(ff_out)
        x = self.ln2(x + ff_out)
        
        return x[-1:] # Extract updated reasoning token

class LRTEnsemble(nn.Module):
    """Ensemble of LRT models for improved accuracy"""
    config: Dict[str, Any]
    num_models: int = 3
    
    def setup(self):
        self.models = [UltraFastLRT(self.config) for _ in range(self.num_models)]
    
    def __call__(self, board_state, **kwargs):
        results = [model(board_state, **kwargs) for model in self.models]
        
        # Average values
        avg_value = jnp.mean(jnp.array([r['value'] for r in results]))
        
        # Average policies
        avg_policy = jnp.mean(jnp.stack([r['policy'] for r in results]), axis=0)
        
        return {
            'value': avg_value,
            'policy': avg_policy,
            'stats': results[0]['stats']  # Use stats from first model
        }
