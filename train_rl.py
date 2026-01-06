import os
import sys
import argparse

# --- JAX STABILITY FLAGS ---
# Force JAX to be more conservative with memory to avoid CUDA_ERROR_ILLEGAL_ADDRESS
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1" # Stability over speed

# Add repo/python to sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if os.path.join(repo_root, "python") not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "python"))
import jax
import jax.numpy as jnp
import numpy as np
from liquid_chess.training.trainer import LRTTrainer
from liquid_chess.models.lrt.complete_model import UltraFastLRT

class RLTrainer(LRTTrainer):
    """Trainer specialized for Reinforcement Learning data."""
    def compute_loss(self, params, batch, rng):
        # Forward pass
        batch_size = batch['outcome'].shape[0]
        dropout_rngs = jax.random.split(rng, batch_size)

        def apply_one(board, dropout_rng):
            return self.model.apply(
                {'params': params},
                board,
                max_steps=self.config['model'].get('max_steps', 16), # Reduce unrolling depth
                deterministic=False,
                rngs={'dropout': dropout_rng},
            )

        outputs = jax.vmap(apply_one, in_axes=(0, 0))(batch['board'], dropout_rngs)

        # RL Targets: 
        # Value targets are z (game outcome)
        # Policy targets are pi (MCTS search probabilities)
        
        # Value loss (MSE)
        value_loss = jnp.mean((outputs['value'] - batch['outcome']) ** 2)
        
        # Policy loss (Cross-Entropy with MCTS policy)
        # pi is the target policy from search
        target_policy = batch['policy'].reshape(batch_size, -1)
        pred_policy = outputs['policy'].reshape(batch_size, -1)
        
        # Add epsilon to prevent log(0)
        policy_loss = -jnp.mean(jnp.sum(target_policy * jnp.log(pred_policy + 1e-10), axis=-1))
        
        # Step penalty
        steps = outputs['stats']['steps_taken']
        step_penalty = self.config['training'].get('step_penalty', 0.01)
        reasoning_loss = step_penalty * jnp.mean(steps)
        
        total_loss = value_loss + policy_loss + reasoning_loss
        
        metrics = {
            'loss': total_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'reasoning_loss': reasoning_loss,
            'steps': jnp.mean(steps),
            'accuracy': jnp.mean(jnp.argmax(pred_policy, axis=-1) == jnp.argmax(target_policy, axis=-1))
        }
        
        return total_loss, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to self-play .npz file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_rl")
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    config = {
        'seed': 42,
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_heads': 8,
            'max_steps': args.max_steps,
            'min_reasoning_steps': 2,
            'dropout_rate': 0.1,
            'use_enhanced_encoder': True,
        },
        'training': {
            'learning_rate': args.lr,
            'end_learning_rate': args.lr / 10,
            'warmup_steps': 100,
            'total_steps': 1000, # Will be adjusted
            'steps_per_epoch': 100,
            'val_steps': 5,
            'batch_size': args.batch_size,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01,
            'value_weight': 1.0,
            'policy_weight': 1.0,
            'step_penalty': 0.005,
            'checkpoint_dir': args.checkpoint_dir,
            'save_every': 1,
            'keep_checkpoints': 5,
        },
        'logging': {'use_wandb': False},
        'resume': args.resume
    }

    # Load dataset
    # Replay buffer: use multiple recent files if available
    import glob
    if "*" in args.data_path:
        all_files = sorted(glob.glob(args.data_path))
        # Sliding Window: Keep the most recent 100 files (approx 100k-150k positions)
        if len(all_files) > 1000:
            print(f"Buffer full ({len(all_files)} files). Using most recent 100.")
            all_files = all_files[-1000:]
    else:
        all_files = [args.data_path]
        
    print(f"Loading data from {len(all_files)} files (Sliding Window Replay Buffer)...")
    
    all_fens = []
    all_policies = []
    all_outcomes = []
    
    for f in all_files:
        data = np.load(f)
        all_fens.extend(data['fens'])
        all_policies.extend(data['policies'])
        all_outcomes.extend(data['outcomes'])
        
    class SimpleDataset:
        def __init__(self, fens, policies, outcomes, batch_size):
            self.fens = np.array(fens)
            self.policies = np.array(policies)
            self.outcomes = np.array(outcomes)
            self.batch_size = batch_size
            self.indices = np.arange(len(self.fens))
            print(f"Dataset initialized with {len(self.fens)} positions.")
            
        def get_batch(self):
            idx = np.random.choice(self.indices, self.batch_size)
            batch_fens = self.fens[idx]
            
            # Use CachedChessDataset's board encoding logic
            from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input
            import chess
            
            # This is slow, but works for a prototype
            boards = [board_to_enhanced_input(chess.Board(fen)) for fen in batch_fens]
            
            batch_board = {}
            for key in boards[0].keys():
                batch_board[key] = jnp.stack([b[key] for b in boards])
            
            return {
                'board': batch_board,
                'policy': jnp.array(self.policies[idx]),
                'outcome': jnp.array(self.outcomes[idx])
            }

    dataset = SimpleDataset(all_fens, all_policies, all_outcomes, args.batch_size)
    trainer = RLTrainer(config)
    
    if args.resume:
        trainer.resume_from_checkpoint()
        
    trainer.train(dataset, dataset, num_epochs=args.epochs)

if __name__ == "__main__":
    main()
