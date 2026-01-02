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
        self.gates = AdaptiveGates(self.config['hidden_dim'])
    
    @nn.compact
    def __call__(self, 
                 board_state: Dict[str, jnp.ndarray],
                 max_steps: int = 50,
                 deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        
        hidden_dim = self.config['hidden_dim']
        use_enhanced_encoder = self.config.get('use_enhanced_encoder', False)

        if use_enhanced_encoder:
            board_emb = EnhancedChessBoardEncoder(hidden_dim)(board_state)
        else:
            # Simple board encoding - just flatten and project
            # pieces: [8, 8] -> [64]
            pieces_flat = board_state['pieces'].flatten()
            pieces_emb = nn.Embed(13, hidden_dim // 4)(pieces_flat)  # [64, hidden_dim//4]
            
            # Simple positional encoding
            pos_enc = nn.Dense(hidden_dim // 4)(pieces_emb)  # [64, hidden_dim//4]
            
            # Combine and project
            board_emb = nn.Dense(hidden_dim)(jnp.concatenate([pieces_emb, pos_enc], axis=-1))  # [64, hidden_dim]
            board_emb = nn.LayerNorm()(board_emb)
        
        # Initialize reasoning token
        init_token = self.param('init_token',
                               random.normal,
                               (1, hidden_dim))
        
        # Create attention module ONCE outside the loop
        # Use deterministic flag properly for dropout
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config['num_heads'],
            qkv_features=hidden_dim,
            dropout_rate=self.config.get('dropout_rate', 0.0),
            deterministic=deterministic,  # Pass through deterministic flag
        )

        complexity = self.gates.estimate_complexity(board_emb)
        adaptive_max_steps = self.gates.compute_adaptive_steps(
            complexity, 
            min_steps=4, 
            max_steps=max_steps
        )
        
        # Simple reasoning: apply transformer iterations
        current_token = init_token
        step = 0
        total_keep_prob = 0.0
        
        while step < adaptive_max_steps:
            # Transformer step
            new_token = self.transformer_step(board_emb, current_token)
            
            # Apply discard gate
            current_token, keep_prob = self.gates.apply_discard_gate(
                current_token, new_token, deterministic
            )
            
            total_keep_prob += keep_prob
            
            # Check stop gate
            stop_prob = self.gates.should_stop(current_token, step)
            
            step += 1
            
            # Early stopping
            if stop_prob > 0.95:
                break
        
        # Output heads
        value = nn.Dense(1)(current_token).squeeze()
        policy_logits = nn.Dense(64 * 64)(current_token).squeeze()
        
        # Format outputs
        value_cp = 100 * jnp.tanh(value)
        policy = nn.softmax(policy_logits).reshape(64, 64)
        
        # Collect all statistics
        stats = {
            'steps_taken': max_steps,
            'avg_keep_prob': jnp.float32(1.0),
            'final_stop_prob': jnp.float32(0.0),
        }
        
        return {
            'value': value_cp,
            'policy': policy,
            'stats': stats,
            'final_token': current_token
        }

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
