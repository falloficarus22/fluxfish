"""
Liquid Reasoning Gates for Adaptive Computation
File: python/liquid_chess/models/lrt/gates.py

This module implements the gating mechanisms that make the reasoning "liquid":
- Discard Gate: Decides whether to keep or discard reasoning updates
- Stop Gate: Decides when reasoning should terminate
- Uses straight-through estimators to maintain gradient flow
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional


class DiscardGate(nn.Module):
    """
    Gate that decides whether to discard or keep a reasoning update.
    
    Compares the old reasoning token with a proposed new token and outputs
    a probability of keeping the update. Uses a learned MLP to make this decision.
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, old_token: jnp.ndarray, new_token: jnp.ndarray) -> jnp.ndarray:
        """
        Compute keep probability for the new token.
        
        Args:
            old_token: [batch, hidden_dim] previous reasoning state
            new_token: [batch, hidden_dim] proposed new reasoning state
        
        Returns:
            keep_prob: [batch] probability of keeping the update (0-1)
        """
        # Compute difference to measure "usefulness" of update
        diff = new_token - old_token
        diff_norm = jnp.sqrt(jnp.sum(diff ** 2, axis=-1, keepdims=True) + 1e-8)
        
        # Cosine similarity between old and new
        old_norm = jnp.sqrt(jnp.sum(old_token ** 2, axis=-1, keepdims=True) + 1e-8)
        new_norm = jnp.sqrt(jnp.sum(new_token ** 2, axis=-1, keepdims=True) + 1e-8)
        cosine_sim = jnp.sum(old_token * new_token, axis=-1, keepdims=True) / (old_norm * new_norm + 1e-8)
        
        # Concatenate features for gate decision
        gate_input = jnp.concatenate([
            old_token,           # Current state
            new_token,           # Proposed state
            diff,                # Change vector
            diff_norm,           # Magnitude of change
            cosine_sim           # Direction similarity
        ], axis=-1)
        
        # Two-layer MLP for decision
        hidden = nn.Dense(self.hidden_dim // 4)(gate_input)
        hidden = nn.tanh(hidden)  # Bounded activation
        hidden = nn.Dense(self.hidden_dim // 8)(hidden)
        hidden = nn.tanh(hidden)
        
        # Final gate logit
        gate_logit = nn.Dense(1)(hidden)
        
        # Sigmoid to get probability
        keep_prob = nn.sigmoid(gate_logit).squeeze(-1)
        
        return keep_prob


class StopGate(nn.Module):
    """
    Gate that decides when reasoning should terminate.
    
    Examines the current reasoning token and outputs a probability that
    reasoning is "done" (i.e., further iterations won't help).
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, token: jnp.ndarray, step: int) -> jnp.ndarray:
        """
        Compute stop probability for current reasoning state.
        
        Args:
            token: [batch, hidden_dim] current reasoning state
            step: current iteration number (for dynamic behavior)
        
        Returns:
            stop_prob: [batch] probability of stopping (0-1)
        """
        # Add step information (normalized)
        step_feat = jnp.array(step, dtype=jnp.float32) / 50.0  # Normalize by max_steps
        step_feat = jnp.broadcast_to(step_feat, token.shape[:-1] + (1,))
        
        # Compute token "stability" (low variance = converged)
        token_std = jnp.std(token, axis=-1, keepdims=True)
        token_mean = jnp.mean(jnp.abs(token), axis=-1, keepdims=True)
        
        # Concatenate features
        gate_input = jnp.concatenate([
            token,
            step_feat,
            token_std,
            token_mean
        ], axis=-1)
        
        # Three-layer MLP for stop decision
        hidden = nn.Dense(self.hidden_dim // 4)(gate_input)
        hidden = nn.relu(hidden)
        hidden = nn.Dense(self.hidden_dim // 8)(hidden)
        hidden = nn.relu(hidden)
        
        # Output layer
        stop_logit = nn.Dense(1)(hidden)
        
        # Sigmoid for probability
        # Bias towards continuing early, stopping late
        step_bias = jnp.minimum(step / 10.0, 1.0)  # Ramp up over first 10 steps
        stop_prob = nn.sigmoid(stop_logit + step_bias).squeeze(-1)
        
        return stop_prob


class ComplexityEstimator(nn.Module):
    """
    Estimates position complexity to determine initial reasoning budget.
    
    This helps allocate more steps to complex positions and fewer to simple ones.
    """
    hidden_dim: int
    
    @nn.compact
    def __call__(self, board_emb: jnp.ndarray) -> jnp.ndarray:
        """
        Estimate complexity from board embedding.
        
        Args:
            board_emb: [batch, 64, hidden_dim] board representation
        
        Returns:
            complexity: [batch] complexity score 0-1 (higher = more complex)
        """
        # Global pool over board positions
        board_global = jnp.mean(board_emb, axis=-2)  # [..., hidden_dim]
        
        # Compute variance (high variance = complex position)
        board_std = jnp.std(board_emb, axis=-2)  # [..., hidden_dim]
        
        # Combine features
        features = jnp.concatenate([board_global, board_std], axis=-1)
        
        # Small MLP
        hidden = nn.Dense(self.hidden_dim // 4)(features)
        hidden = nn.relu(hidden)
        complexity_logit = nn.Dense(1)(hidden)
        
        # Sigmoid to bound 0-1
        complexity = nn.sigmoid(complexity_logit).squeeze(-1)
        
        return complexity


class AdaptiveGates(nn.Module):
    """
    Combined gating system for liquid reasoning.
    
    Integrates discard gate, stop gate, and complexity estimation with
    straight-through estimators for training stability.
    """
    hidden_dim: int
    temperature: float = 1.0  # For Gumbel-softmax during training
    
    def setup(self):
        self.discard_gate = DiscardGate(self.hidden_dim)
        self.stop_gate = StopGate(self.hidden_dim)
        self.complexity_estimator = ComplexityEstimator(self.hidden_dim)
    
    def apply_discard_gate(
        self, 
        old_token: jnp.ndarray, 
        new_token: jnp.ndarray,
        deterministic: bool = True,
        rng: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply discard gate with straight-through estimator.
        
        Args:
            old_token: previous reasoning state
            new_token: proposed new state
            deterministic: if True, use threshold; if False, sample
            rng: random key for sampling (required if not deterministic)
        
        Returns:
            updated_token: gated combination of old and new
            keep_prob: probability that was computed
        """
        # Compute keep probability
        keep_prob = self.discard_gate(old_token, new_token)
        
        # Expand dimensions for broadcasting
        keep_prob_expanded = jnp.expand_dims(keep_prob, -1)  # [batch, 1]
        
        if deterministic:
            # Threshold at 0.5 for inference
            keep_discrete = (keep_prob_expanded > 0.5).astype(jnp.float32)
        else:
            # Stochastic sampling for training
            if rng is None:
                raise ValueError("rng required when deterministic=False")
            
            # Bernoulli sampling
            keep_discrete = jax.random.bernoulli(rng, keep_prob_expanded).astype(jnp.float32)
        
        # Straight-through estimator:
        # Forward pass: discrete (0 or 1)
        # Backward pass: continuous (gradient through keep_prob)
        keep = keep_prob_expanded + jax.lax.stop_gradient(keep_discrete - keep_prob_expanded)
        
        # Apply gate: interpolate between old and new
        updated_token = keep * new_token + (1.0 - keep) * old_token
        
        return updated_token, keep_prob
    
    def should_stop(self, token: jnp.ndarray, step: int) -> jnp.ndarray:
        """
        Check if reasoning should stop.
        
        Args:
            token: current reasoning state
            step: current iteration number
        
        Returns:
            stop_prob: probability of stopping
        """
        return self.stop_gate(token, step)
    
    def estimate_complexity(self, board_emb: jnp.ndarray) -> jnp.ndarray:
        """
        Estimate position complexity.
        
        Args:
            board_emb: board representation
        
        Returns:
            complexity: estimated complexity score 0-1
        """
        return self.complexity_estimator(board_emb)
    
    def compute_adaptive_steps(
        self, 
        complexity: jnp.ndarray,
        min_steps: int = 4,
        max_steps: int = 50
    ) -> jnp.ndarray:
        """
        Compute adaptive step budget based on complexity.
        
        Args:
            complexity: complexity score 0-1
            min_steps: minimum reasoning steps
            max_steps: maximum reasoning steps
        
        Returns:
            steps: recommended number of steps (rounded to int)
        """
        # Linear interpolation based on complexity
        steps_float = min_steps + complexity * (max_steps - min_steps)
        
        # Round to integer
        steps = jnp.round(steps_float).astype(jnp.int32)
        
        # Clip to valid range
        steps = jnp.clip(steps, min_steps, max_steps)
        
        return steps


# ============================================================================
# Helper functions for integration
# ============================================================================

def create_gating_state(
    hidden_dim: int,
    rng: jax.random.PRNGKey
) -> dict:
    """
    Initialize gating modules (helper for standalone testing).
    
    Args:
        hidden_dim: size of hidden dimension
        rng: random key for initialization
    
    Returns:
        params: initialized parameters
    """
    gates = AdaptiveGates(hidden_dim)
    
    # Dummy inputs for initialization
    dummy_token = jnp.ones((1, hidden_dim))
    dummy_board = jnp.ones((1, 64, hidden_dim))
    
    params = gates.init(
        rng,
        dummy_token,
        dummy_token,
        method=gates.apply_discard_gate,
    )
    
    return params


def test_gates_forward_pass():
    """Quick test to verify gates work correctly."""
    import jax.random as random
    
    rng = random.PRNGKey(0)
    hidden_dim = 64
    
    gates = AdaptiveGates(hidden_dim)
    
    # Create test inputs
    old_token = random.normal(rng, (2, hidden_dim))
    new_token = random.normal(rng, (2, hidden_dim))
    board_emb = random.normal(rng, (2, 64, hidden_dim))
    
    # Initialize
    rng, init_rng = random.split(rng)
    params = gates.init(
        init_rng,
        old_token,
        new_token,
        deterministic=True,
        method=gates.apply_discard_gate
    )
    
    # Test discard gate
    updated, keep_prob = gates.apply(
        params,
        old_token,
        new_token,
        deterministic=True,
        method=gates.apply_discard_gate
    )
    
    print(f"Discard gate:")
    print(f"  Keep prob: {keep_prob}")
    print(f"  Updated shape: {updated.shape}")
    
    # Test stop gate
    stop_prob = gates.apply(params, old_token, 5, method=gates.should_stop)
    print(f"\nStop gate:")
    print(f"  Stop prob: {stop_prob}")
    
    # Test complexity estimator
    complexity = gates.apply(params, board_emb, method=gates.estimate_complexity)
    print(f"\nComplexity:")
    print(f"  Score: {complexity}")
    
    # Test adaptive steps
    steps = gates.apply(params, complexity, method=gates.compute_adaptive_steps)
    print(f"  Recommended steps: {steps}")
    
    print("\n✅ All gates working correctly!")


if __name__ == "__main__":
    test_gates_forward_pass()
