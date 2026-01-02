import pytest
import jax
from jax import random
import jax.numpy as jnp

def test_gates_initialization():
    from liquid_chess.models.lrt.gates import AdaptiveGates
    
    gates = AdaptiveGates(hidden_dim=64)
    rng = random.PRNGKey(0)
    
    old_token = jnp.ones((2, 64))
    new_token = jnp.ones((2, 64)) * 2
    
    params = gates.init(rng, old_token, new_token, deterministic=True,
                       method=gates.apply_discard_gate)
    
    updated, keep_prob = gates.apply(params, old_token, new_token, 
                                     deterministic=True,
                                     method=gates.apply_discard_gate)
    
    assert updated.shape == (2, 64)
    assert keep_prob.shape == (2,)
    assert jnp.all((keep_prob >= 0) & (keep_prob <= 1))

def test_gates_gradient_flow():
    from liquid_chess.models.lrt.gates import AdaptiveGates
    
    gates = AdaptiveGates(hidden_dim=64)
    rng = random.PRNGKey(0)
    
    k1, k2 = random.split(rng)
    old_token = random.normal(k1, (2, 64))
    new_token = random.normal(k2, (2, 64))
    
    params = gates.init(rng, old_token, new_token, deterministic=True,
                       method=gates.apply_discard_gate)
    
    def loss_fn(params):
        updated, _ = gates.apply(params, old_token, new_token,
                                deterministic=True,
                                method=gates.apply_discard_gate)
        return jnp.sum(updated ** 2)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)
    
    # Verify gradients exist and are non-zero
    grad_vals = jax.tree_util.tree_leaves(grads)
    total_grad = sum(jnp.sum(jnp.abs(g)) for g in grad_vals)
    
    assert total_grad > 0, "No gradients flowing!"

if __name__ == "__main__":
    test_gates_initialization()
    test_gates_gradient_flow()
    print("✅ All gate tests passed!")