import jax
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
import optax
import flax
from flax.training import train_state, checkpoints
import tensorflow as tf
import numpy as np
from typing import Any, Dict, Tuple, Optional
import wandb
from tqdm import tqdm
import chess
import chess.pgn

class LRTTrainer:
    """Training pipeline for Liquid Reasoning Transformer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = random.PRNGKey(config.get('seed', 42))
        
        # Initialize model
        self.model = UltraFastLRT(config['model'])
        self.state = None
        
        # Create optimizer
        self.schedule = self._create_learning_rate_schedule()
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config['training']['max_grad_norm']),
            optax.adamw(learning_rate=self.schedule, 
                       weight_decay=config['training']['weight_decay'])
        )
        
        # Initialize metrics
        self.metrics = {}
        
    def _create_learning_rate_schedule(self):
        """Create learning rate schedule with warmup and decay"""
        training_cfg = self.config['training']
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=training_cfg['learning_rate'],
            warmup_steps=training_cfg['warmup_steps'],
            decay_steps=training_cfg['total_steps'],
            end_value=training_cfg['end_learning_rate']
        )
        
        return schedule
    
    def create_train_state(self) -> train_state.TrainState:
        """Initialize training state"""
        # Sample input
        sample_input = self._create_sample_input()
        
        # Initialize model
        rng, init_rng = random.split(self.rng)
        params = self.model.init(init_rng, sample_input)['params']
        
        # Create state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer
        )
        
        return state
    
    def _create_sample_input(self) -> Dict[str, jnp.ndarray]:
        """Create sample input for initialization"""
        return {
            'pieces': jnp.zeros((8, 8), dtype=jnp.int8),
            'turn': jnp.array(True, dtype=jnp.bool_),
            'castling': jnp.zeros((4,), dtype=jnp.bool_),
            'ep_square': jnp.array(-1, dtype=jnp.int8)
        }
    
    def compute_loss(self, params, batch, rng) -> Tuple[jnp.ndarray, Dict]:
        """Compute loss for a batch"""
        # Forward pass
        outputs = self.model.apply({'params': params}, batch['board'])
        
        # Value loss (MSE with game outcome)
        value_loss = jnp.mean((outputs['value'] - batch['outcome']) ** 2)
        
        # Policy loss (cross-entropy with search probabilities)
        policy_target = batch['policy']
        policy_pred = outputs['policy']
        
        # Flatten for cross-entropy
        policy_target_flat = policy_target.reshape(-1)
        policy_pred_flat = policy_pred.reshape(-1)
        
        # Mask illegal moves
        legal_mask = batch['legal_moves'].reshape(-1)
        policy_target_masked = policy_target_flat * legal_mask
        policy_pred_masked = policy_pred_flat * legal_mask
        
        # Normalize
        policy_target_masked = policy_target_masked / (jnp.sum(policy_target_masked) + 1e-10)
        
        # Cross-entropy loss
        policy_loss = -jnp.sum(policy_target_masked * jnp.log(policy_pred_masked + 1e-10))
        
        # Regularization: encourage efficient reasoning
        steps = outputs['stats']['steps_taken']
        step_penalty = self.config['training'].get('step_penalty', 0.01)
        reasoning_loss = step_penalty * steps
        
        # Total loss
        total_loss = (
            self.config['training']['value_weight'] * value_loss +
            self.config['training']['policy_weight'] * policy_loss +
            reasoning_loss
        )
        
        # Metrics
        metrics = {
            'loss': total_loss,
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'reasoning_loss': reasoning_loss,
            'accuracy': jnp.mean(jnp.argmax(policy_pred, axis=-1) == 
                                jnp.argmax(policy_target, axis=-1)),
            'steps': steps,
            'value_mse': value_loss
        }
        
        return total_loss, metrics
    
    @partial(jit, static_argnums=(0,))
    def train_step(self, state, batch, rng):
        """Single training step"""
        
        def loss_fn(params):
            return self.compute_loss(params, batch, rng)
        
        # Compute gradient
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, metrics
    
    def train(self, train_dataset, val_dataset, num_epochs: int):
        """Main training loop"""
        
        # Initialize state
        if self.state is None:
            self.state = self.create_train_state()
        
        # Initialize wandb
        if self.config['logging'].get('use_wandb', False):
            wandb.init(project="liquid-chess", config=self.config)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            train_metrics = self._train_epoch(train_dataset, epoch)
            
            # Validation
            val_metrics = self._validate_epoch(val_dataset)
            
            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self._save_checkpoint(epoch)
        
        # Final save
        self._save_checkpoint('final')
        
        if self.config['logging'].get('use_wandb', False):
            wandb.finish()
    
    def _train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        metrics_history = []
        
        # Create progress bar
        pbar = tqdm(dataset, desc=f"Epoch {epoch}", 
                   total=self.config['training']['steps_per_epoch'])
        
        for step, batch in enumerate(pbar):
            if step >= self.config['training']['steps_per_epoch']:
                break
            
            # Update RNG
            self.rng, step_rng = random.split(self.rng)
            
            # Training step
            self.state, metrics = self.train_step(self.state, batch, step_rng)
            
            # Update metrics
            metrics_history.append(metrics)
            
            # Update progress bar
            if step % 10 == 0:
                avg_metrics = {
                    k: np.mean([m[k] for m in metrics_history[-10:]])
                    for k in metrics.keys()
                }
                pbar.set_postfix(avg_metrics)
        
        # Compute epoch averages
        epoch_metrics = {
            k: np.mean([m[k] for m in metrics_history])
            for k in metrics_history[0].keys()
        }
        
        return epoch_metrics
    
    def _validate_epoch(self, dataset) -> Dict[str, float]:
        """Validate for one epoch"""
        all_metrics = []
        
        for batch in dataset:
            # Forward pass
            outputs = self.model.apply({'params': self.state.params}, 
                                      batch['board'])
            
            # Compute metrics
            _, metrics = self.compute_loss(self.state.params, batch, self.rng)
            all_metrics.append(metrics)
        
        # Average metrics
        val_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        return val_metrics
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to wandb and console"""
        
        # Console logging
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Avg Steps: {train_metrics['steps']:.2f}")
        
        # Wandb logging
        if self.config['logging'].get('use_wandb', False):
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/value_loss': train_metrics['value_loss'],
                'train/policy_loss': train_metrics['policy_loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/steps': train_metrics['steps'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/steps': val_metrics['steps'],
                'lr': self.schedule(self.state.step)
            }
            wandb.log(log_dict)
    
    def _save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoints.save_checkpoint(
            ckpt_dir=self.config['training']['checkpoint_dir'],
            target=self.state,
            step=step,
            keep=self.config['training'].get('keep_checkpoints', 3)
        )
    
    def export_model(self, path: str):
        """Export model for inference"""
        # Convert to inference format
        inference_fn = lambda x: self.model.apply({'params': self.state.params}, x)
        
        # Save
        with open(path, 'wb') as f:
            # In practice, you'd use a proper serialization format
            # This is simplified
            import pickle
            pickle.dump({
                'params': self.state.params,
                'config': self.config
            }, f)

class ChessDataset:
    """Dataset for training LRT on chess positions"""
    
    def __init__(self, data_paths, batch_size: int = 32, shuffle: bool = True):
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load and preprocess data
        self.positions = self._load_positions()
        
    def _load_positions(self):
        """Load chess positions from files"""
        positions = []
        
        for path in self.data_paths:
            if path.endswith('.pgn'):
                positions.extend(self._load_pgn(path))
            elif path.endswith('.bin'):
                positions.extend(self._load_bin(path))
        
        return positions
    
    def _load_pgn(self, pgn_path: str):
        """Load positions from PGN file"""
        positions = []
        
        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                # Extract positions from game
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    
                    # Convert to training example
                    example = self._board_to_example(board, game.headers['Result'])
                    positions.append(example)
        
        return positions
    
    def _board_to_example(self, board: chess.Board, result: str) -> Dict:
        """Convert chess board to training example"""
        
        # Board representation
        pieces = np.zeros((8, 8), dtype=np.int8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Map piece to index (1-12)
                idx = (piece.piece_type - 1) * 2 + (0 if piece.color == chess.WHITE else 1)
                pieces[chess.square_rank(square), chess.square_file(square)] = idx
        
        # Outcome
        if result == '1-0':
            outcome = 1.0  # White wins
        elif result == '0-1':
            outcome = -1.0  # Black wins
        else:
            outcome = 0.0  # Draw
        
        # Legal moves (for policy target)
        legal_moves = np.zeros((64, 64), dtype=np.float32)
        for move in board.legal_moves:
            from_idx = move.from_square
            to_idx = move.to_square
            legal_moves[from_idx, to_idx] = 1.0
        
        # Normalize
        if legal_moves.sum() > 0:
            legal_moves = legal_moves / legal_moves.sum()
        
        return {
            'pieces': pieces,
            'turn': np.array(board.turn, dtype=np.bool_),
            'castling': np.array([
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)
            ], dtype=np.bool_),
            'ep_square': np.array(
                board.ep_square if board.ep_square else -1,
                dtype=np.int8
            ),
            'outcome': np.array(outcome, dtype=np.float32),
            'policy': legal_moves,
            'legal_moves': (legal_moves > 0).astype(np.float32)
        }
    
    def get_batch(self) -> Dict[str, np.ndarray]:
        """Get a batch of training data"""
        if self.shuffle:
            indices = np.random.choice(len(self.positions), self.batch_size, replace=True)
        else:
            # Sequential batching
            pass
        
        batch = {k: [] for k in self.positions[0].keys()}
        
        for idx in indices:
            example = self.positions[idx]
            for k, v in example.items():
                batch[k].append(v)
        
        # Stack arrays
        for k in batch.keys():
            batch[k] = np.stack(batch[k])
        
        return batch