import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any
import chex
from functools import partial

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

class LiquidReasoningCell(nn.Module):
    """Single step of liquid reasoning"""
    config: Dict[str, Any]
    
    @nn.compact
    def __call__(self, 
                 board_emb: jnp.ndarray,        # [64, hidden]
                 reasoning_token: jnp.ndarray,  # [1, hidden]
                 step: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        # 1. Attention between board and reasoning token
        # Concatenate token to board
        all_tokens = jnp.concatenate([board_emb, reasoning_token], axis=0)
        
        # Multi-head attention
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.config['num_heads'],
            qkv_features=self.config['hidden_dim'],
            dropout_rate=self.config.get('dropout_rate', 0.0)
        )(all_tokens, all_tokens)
        
        # Extract updated reasoning token
        updated_token = attended[-1:]  # Last token
        
        # 2. Discard gate - decide if update is valuable
        discard_input = jnp.concatenate([
            reasoning_token.flatten(),
            updated_token.flatten()
        ])
        
        discard_prob = nn.Dense(1)(discard_input)
        discard_prob = nn.sigmoid(discard_prob)[0]
        
        # 3. Consistency check - ensure legal board state
        legal_prob = self.consistency_check(board_emb, updated_token)
        
        # 4. Combine gate decisions
        keep_prob = (1 - discard_prob) * legal_prob
        
        # 5. Always deterministic during loop (no dropout RNG)
        final_token = jnp.where(keep_prob > 0.5, updated_token, reasoning_token)
        
        return final_token, keep_prob
    
    def consistency_check(self, board_emb: jnp.ndarray, 
                         token: jnp.ndarray) -> jnp.ndarray:
        """Check if reasoning maintains legal board constraints"""
        # Predict basic board properties from token
        pred_turn = nn.Dense(1)(token)  # Whose turn
        pred_in_check = nn.Dense(2)(token)  # Is each side in check?
        
        # Compare with actual board (computed from board_emb)
        # This is a simplified check - real implementation would be more complex
        consistency = nn.sigmoid(pred_in_check.mean())
        
        return consistency

class AdaptiveStopGate(nn.Module):
    """Decide when to stop reasoning"""
    config: Dict[str, Any]
    
    @nn.compact
    def __call__(self, 
                 reasoning_token: jnp.ndarray,
                 prev_token: jnp.ndarray,
                 step: int) -> jnp.ndarray:
        
        # Features for stop decision
        token_diff = jnp.abs(reasoning_token - prev_token).mean()
        token_std = jnp.std(reasoning_token)
        
        # Step-based decay
        step_factor = jnp.exp(-step / 10.0)
        
        # Network to predict stop probability
        features = jnp.concatenate([
            reasoning_token.flatten(),
            jnp.array([token_diff, token_std, step, step_factor])
        ])
        
        stop_logit = nn.Dense(128)(features)
        stop_logit = nn.relu(stop_logit)
        stop_logit = nn.Dense(1)(stop_logit)
        
        stop_prob = nn.sigmoid(stop_logit)[0]
        
        # Never stop before minimum steps
        min_steps = self.config.get('min_reasoning_steps', 2)
        stop_prob = jnp.where(step < min_steps, 0.0, stop_prob)
        
        return stop_prob

class UltraFastLRT(nn.Module):
    """Complete Liquid Reasoning Transformer for chess"""
    config: Dict[str, Any]
    
    def setup(self):
        # Components
        self.encoder = ChessBoardEncoder(features=self.config['hidden_dim'])
        self.reasoning_cell = LiquidReasoningCell(self.config)
        self.stop_gate = AdaptiveStopGate(self.config)
        
        # Initial reasoning token (learned)
        self.init_token = self.param('init_token',
                                    random.normal,
                                    (1, self.config['hidden_dim']))
        
        # Evaluation heads
        self.value_head = nn.Dense(1)
        self.policy_head = nn.Dense(64 * 64)  # All possible moves
    
    @nn.compact
    def __call__(self, 
                 board_state: Dict[str, jnp.ndarray],
                 max_steps: int = 50,
                 deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        
        # Encode board
        board_emb = self.encoder(board_state)
        
        # Initialize reasoning
        reasoning_token = self.init_token

        # Adaptive reasoning loop (differentiable fori_loop)
        def body_fn(step, carry):
            final_token, keep_sum = carry
            # Apply reasoning cell
            new_token, keep_prob = self.reasoning_cell(
                board_emb,
                final_token,
                step,
            )
            stop_prob = self.stop_gate(new_token, final_token, step)
            stop = stop_prob > 0.95
            # Update for next iteration
            final_token = jnp.where(stop, final_token, new_token)
            keep_sum = keep_sum + keep_prob
            return (final_token, keep_sum)

        init_carry = (reasoning_token, jnp.float32(0.0))
        final_token, keep_sum = lax.fori_loop(0, max_steps, body_fn, init_carry)
        steps_taken = max_steps
        avg_keep_prob = keep_sum / max_steps
        
        # Generate outputs
        value = self.value_head(final_token).squeeze()
        policy_logits = self.policy_head(final_token)
        policy_probs = nn.softmax(policy_logits, axis=-1)
        policy = policy_probs.reshape(64, 64)
        
        # Apply legal move mask (optional - can be done in C++)
        # policy = policy * legal_moves_mask
        
        # Convert value to centipawns
        value_cp = 100 * jnp.tanh(value)
        
        # Collect all statistics
        stats = {
            'steps_taken': steps_taken,
            'avg_keep_prob': avg_keep_prob,
        }
        
        return {
            'value': value_cp,
            'policy': policy,
            'stats': stats,
            'final_token': final_token
        }
    
    @partial(jit, static_argnums=(0, 3, 4))
    def evaluate_fast(self, 
                     board_state: Dict[str, jnp.ndarray],
                     cache_key: Optional[jnp.ndarray] = None,
                     max_steps: int = 50,
                     deterministic: bool = True) -> Tuple[jnp.ndarray, Dict]:
        """
        Optimized evaluation for integration with C++ engine.
        
        Args:
            board_state: Chess position
            cache_key: Optional cache key for memoization
            max_steps: Maximum reasoning steps
            deterministic: Whether to use deterministic gates
            
        Returns:
            value: Position evaluation
            metadata: Additional information
        """
        # Simple caching mechanism (could be expanded)
        if cache_key is not None:
            # In practice, you'd use a proper cache here
            pass
        
        # Run LRT
        result = self.__call__(board_state, max_steps, deterministic)
        
        # Extract value and policy
        value = result['value']
        policy = result['policy']
        stats = result['stats']
        
        # Get top moves from policy
        policy_flat = policy.flatten()
        topk_values, topk_indices = jax.lax.top_k(policy_flat, 5)
        
        # Convert indices to move coordinates
        top_moves = []
        for idx, prob in zip(topk_indices, topk_values):
            from_sq = idx // 64
            to_sq = idx % 64
            top_moves.append({
                'from': int(from_sq),
                'to': int(to_sq),
                'probability': float(prob)
            })
        
        metadata = {
            'top_moves': top_moves,
            'reasoning_steps': int(stats['steps_taken']),
            'complexity': float(1.0 - stats['avg_keep_prob']),
            'certainty': float(1.0 - stats['final_stop_prob'])
        }
        
        return value, metadata

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
        
        # Ensemble policy (geometric mean)
        policies = jnp.stack([r['policy'] for r in results])
        avg_policy = jnp.exp(jnp.mean(jnp.log(policies + 1e-10), axis=0))
        avg_policy = avg_policy / jnp.sum(avg_policy)  # Renormalize
        
        # Aggregate statistics
        avg_steps = jnp.mean(jnp.array([r['stats']['steps_taken'] for r in results]))
        
        return {
            'value': avg_value,
            'policy': avg_policy,
            'stats': {'steps_taken': avg_steps}
        }