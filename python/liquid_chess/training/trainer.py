import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import flax
from flax.training import train_state, checkpoints
import numpy as np
from typing import Any, Dict, Tuple, Optional
from functools import partial
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None
from tqdm import tqdm
import chess
import chess.pgn

from liquid_chess.models.lrt.complete_model import UltraFastLRT
from liquid_chess.models.lrt.feature_extraction import board_to_enhanced_input

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
        
        warmup_steps = training_cfg['warmup_steps']
        # In newer optax versions, decay_steps is the total steps (end step), not duration.
        # We ensure it's at least warmup_steps + 1 to avoid negative/zero decay duration.
        total_steps = max(warmup_steps + 1, training_cfg['total_steps'])
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=training_cfg['learning_rate'],
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
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

    def resume_from_checkpoint(self):
        """Resume training from latest checkpoint"""
        if self.state is None:
            self.state = self.create_train_state()
            
        checkpoint_dir = self.config['training']['checkpoint_dir']
        print(f"Checking for checkpoints in {checkpoint_dir}...")
        
        # Orbax/Flax often requires absolute paths
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        
        self.state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=self.state
        )
        print(f"Resumed from step {self.state.step}")
    
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
        # Forward pass (vmap over batch)
        batch_size = batch['outcome'].shape[0]
        dropout_rngs = random.split(rng, batch_size)

        def apply_one(board, dropout_rng):
            return self.model.apply(
                {'params': params},
                board,
                max_steps=self.config['model'].get('max_steps', 50),
                deterministic=self.config['model'].get('deterministic', True),
                rngs={'dropout': dropout_rng},
            )

        outputs = jax.vmap(apply_one, in_axes=(0, 0))(batch['board'], dropout_rngs)
        
        # Value loss (MSE with game outcome)
        value_loss = jnp.mean((outputs['value'] - batch['outcome']) ** 2)
        
        # Policy loss (cross-entropy over all moves)
        policy_target = batch['policy'].reshape(batch_size, -1)
        policy_pred = outputs['policy'].reshape(batch_size, -1)
        legal_mask = batch['legal_moves'].reshape(batch_size, -1)

        # Mask illegal moves
        policy_target = policy_target * legal_mask
        policy_pred = policy_pred * legal_mask

        # Normalize targets and predictions within legal moves
        policy_target = policy_target / (jnp.sum(policy_target, axis=-1, keepdims=True) + 1e-10)
        policy_pred = policy_pred / (jnp.sum(policy_pred, axis=-1, keepdims=True) + 1e-10)

        # Cross-entropy loss (mean over batch)
        policy_loss = -jnp.mean(jnp.sum(policy_target * jnp.log(policy_pred + 1e-10), axis=-1))
        
        # Regularization: encourage efficient reasoning
        steps = outputs['stats']['steps_taken']
        step_penalty = self.config['training'].get('step_penalty', 0.01)
        reasoning_loss = step_penalty * jnp.mean(steps)
        
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
            'steps': jnp.mean(steps),
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
        
        # Initialize state if not already done (by resume_from_checkpoint)
        if self.state is None:
            self.state = self.create_train_state()
        
        # Initialize wandb
        if self.config['logging'].get('use_wandb', False):
            if wandb is None:
                raise RuntimeError("Weights & Biases logging is enabled but 'wandb' is not installed")
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
        self._save_checkpoint(num_epochs)
        
        if self.config['logging'].get('use_wandb', False):
            wandb.finish()
    
    def _train_epoch(self, dataset, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        metrics_history = []
        
        # Create progress bar
        pbar = tqdm(range(self.config['training']['steps_per_epoch']), desc=f"Epoch {epoch}")
        
        for step in pbar:
            batch = dataset.get_batch()
            
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
        
        for _ in range(self.config['training'].get('val_steps', 10)):
            batch = dataset.get_batch()
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
    
    def __init__(self, data_paths, batch_size: int = 32, shuffle: bool = True, use_enhanced_features: bool = False):
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_enhanced_features = use_enhanced_features
        
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

    def _load_bin(self, bin_path: str):
        raise NotImplementedError(".bin dataset format is not implemented; please provide .pgn files")
    
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
        
        # Use enhanced feature extraction if configured
        if self.use_enhanced_features:
            board_input = board_to_enhanced_input(board)
        else:
            # Original simple extraction
            pieces = np.zeros((8, 8), dtype=np.int8)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    idx = (piece.piece_type - 1) * 2 + (1 if piece.color == chess.WHITE else 2)
                    pieces[chess.square_rank(square), chess.square_file(square)] = idx
            
            board_input = {
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
            }
        
        # Outcome
        if result == '1-0':
            outcome = 100.0
        elif result == '0-1':
            outcome = -100.0
        else:
            outcome = 0.0
        
        # Legal moves (for policy target)
        legal_moves = np.zeros((64, 64), dtype=np.float32)
        for move in board.legal_moves:
            from_idx = move.from_square
            to_idx = move.to_square
            legal_moves[from_idx, to_idx] = 1.0
        
        if legal_moves.sum() > 0:
            legal_moves = legal_moves / legal_moves.sum()
        
        # Combine board input with training targets
        return {
            **board_input,
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
            if not hasattr(self, '_cursor'):
                self._cursor = 0
            start = self._cursor
            end = start + self.batch_size
            idxs = np.arange(start, end) % len(self.positions)
            indices = idxs
            self._cursor = int(end % len(self.positions))
        
        batch = {k: [] for k in self.positions[0].keys()}
        
        for idx in indices:
            example = self.positions[idx]
            for k, v in example.items():
                batch[k].append(v)
        
        # Stack arrays
        for k in batch.keys():
            batch[k] = np.stack(batch[k])

        board = {
            'pieces': batch['pieces'],
            'turn': batch['turn'],
            'castling': batch['castling'],
            'ep_square': batch['ep_square'],
        }

        out = {
            'board': {k: jnp.array(v) for k, v in board.items()},
            'outcome': jnp.array(batch['outcome']),
            'policy': jnp.array(batch['policy']),
            'legal_moves': jnp.array(batch['legal_moves']),
        }

        return out
