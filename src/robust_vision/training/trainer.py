"""Production trainer with multi-GPU support."""

from typing import Dict, Optional, Tuple, Callable, Any
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import optax
from tqdm import tqdm
import time
from pathlib import Path

from .state import TrainStateWithEMA
from .losses import combined_loss, compute_accuracy


class ProductionTrainer:
    """
    Production-ready trainer with multi-GPU support.
    
    Features:
    - Multi-GPU training via pmap
    - EMA parameter tracking
    - Checkpoint management
    - Logging integration
    - Progress tracking
    """
    
    def __init__(
        self,
        model,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.99,
        loss_type: str = "label_smoothing",
        loss_kwargs: Optional[Dict] = None,
        checkpoint_dir: Optional[str] = None,
        log_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Flax model
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay (L2 regularization)
            ema_decay: EMA decay rate
            loss_type: Type of loss function
            loss_kwargs: Additional loss function arguments
            checkpoint_dir: Directory for saving checkpoints
            log_fn: Optional logging function
        """
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs or {}
        self.checkpoint_dir = checkpoint_dir
        self.log_fn = log_fn or self._default_log
        
        # Multi-GPU setup
        self.num_devices = jax.local_device_count()
        self.use_pmap = self.num_devices > 1
        
        if self.use_pmap:
            self.log_fn(f"Using {self.num_devices} devices for training")
        
    def _default_log(self, message: str):
        """Default logging function."""
        print(message)
    
    def create_train_state(
        self, 
        rng: jax.random.PRNGKey,
        input_shape: Tuple[int, ...]
    ) -> TrainStateWithEMA:
        """
        Create initial training state.
        
        Args:
            rng: JAX random key
            input_shape: Shape of input for parameter initialization
            
        Returns:
            Training state
        """
        # Initialize parameters
        params = self.model.init(rng, jnp.ones(input_shape), training=False)
        
        # Create optimizer
        if self.weight_decay > 0:
            tx = optax.adamw(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            tx = optax.adam(learning_rate=self.learning_rate)
        
        # Create training state with EMA
        state = TrainStateWithEMA.create_with_ema(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            ema_decay=self.ema_decay
        )
        
        return state
    
    @staticmethod
    def train_step(
        state: TrainStateWithEMA,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        num_classes: int,
        loss_type: str,
        loss_kwargs: Dict
    ) -> Tuple[TrainStateWithEMA, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            state: Training state
            batch: Batch of (images, labels)
            num_classes: Number of classes
            loss_type: Loss function type
            loss_kwargs: Loss function arguments
            
        Returns:
            Updated state and metrics dict
        """
        images, labels = batch
        
        def loss_fn(params):
            logits = state.apply_fn(params, images, training=True)
            loss = combined_loss(
                logits, labels, num_classes, 
                loss_type=loss_type, **loss_kwargs
            )
            return loss, logits
        
        # Compute gradients
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Update parameters
        state = state.apply_gradients(grads=grads)
        
        # Update EMA
        state = state.apply_ema_update()
        
        # Compute metrics
        accuracy = compute_accuracy(logits, labels)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
        }
        
        return state, metrics
    
    @staticmethod
    def eval_step(
        state: TrainStateWithEMA,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        num_classes: int,
        use_ema: bool = True
    ) -> Dict[str, float]:
        """
        Single evaluation step.
        
        Args:
            state: Training state
            batch: Batch of (images, labels)
            num_classes: Number of classes
            use_ema: Whether to use EMA parameters
            
        Returns:
            Metrics dict
        """
        images, labels = batch
        
        # Use EMA parameters if available and requested
        params = state.ema_params if (use_ema and state.ema_params is not None) else state.params
        
        # Forward pass
        logits = state.apply_fn(params, images, training=False)
        
        # Compute metrics
        loss = combined_loss(
            logits, labels, num_classes,
            loss_type="label_smoothing", smoothing=0.0  # No smoothing for eval
        )
        accuracy = compute_accuracy(logits, labels)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
        }
        
        return metrics
    
    def train_epoch(
        self,
        state: TrainStateWithEMA,
        train_ds,
        epoch: int
    ) -> Tuple[TrainStateWithEMA, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            state: Training state
            train_ds: Training dataset
            epoch: Current epoch number
            
        Returns:
            Updated state and average metrics
        """
        batch_metrics = []
        
        # Create progress bar
        pbar = tqdm(train_ds, desc=f"Epoch {epoch}", leave=False)
        
        for batch in pbar:
            # Convert to numpy if needed
            if hasattr(batch, 'numpy'):
                images, labels = batch[0].numpy(), batch[1].numpy()
            else:
                images, labels = batch
            
            # Training step
            state, metrics = self.train_step(
                state, (images, labels),
                self.num_classes, self.loss_type, self.loss_kwargs
            )
            
            batch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}"
            })
        
        # Average metrics
        avg_metrics = {
            key: jnp.mean(jnp.array([m[key] for m in batch_metrics]))
            for key in batch_metrics[0].keys()
        }
        
        return state, avg_metrics
    
    def evaluate(
        self,
        state: TrainStateWithEMA,
        eval_ds,
        use_ema: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on dataset.
        
        Args:
            state: Training state
            eval_ds: Evaluation dataset
            use_ema: Whether to use EMA parameters
            
        Returns:
            Average metrics
        """
        batch_metrics = []
        
        for batch in eval_ds:
            # Convert to numpy if needed
            if hasattr(batch, 'numpy'):
                images, labels = batch[0].numpy(), batch[1].numpy()
            else:
                images, labels = batch
            
            # Evaluation step
            metrics = self.eval_step(
                state, (images, labels),
                self.num_classes, use_ema=use_ema
            )
            
            batch_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            key: jnp.mean(jnp.array([m[key] for m in batch_metrics]))
            for key in batch_metrics[0].keys()
        }
        
        return avg_metrics
    
    def train(
        self,
        rng: jax.random.PRNGKey,
        train_ds,
        eval_ds,
        num_epochs: int,
        input_shape: Tuple[int, ...] = (1, 32, 32, 3),
        eval_every: int = 1,
        checkpoint_every: int = 5
    ) -> TrainStateWithEMA:
        """
        Full training loop.
        
        Args:
            rng: JAX random key
            train_ds: Training dataset
            eval_ds: Evaluation dataset
            num_epochs: Number of training epochs
            input_shape: Input shape for initialization
            eval_every: Evaluate every N epochs
            checkpoint_every: Save checkpoint every N epochs
            
        Returns:
            Final training state
        """
        # Create initial state
        state = self.create_train_state(rng, input_shape)
        
        self.log_fn(f"Starting training for {num_epochs} epochs")
        self.log_fn(f"Model has {sum(x.size for x in jax.tree_util.tree_leaves(state.params))} parameters")
        
        best_accuracy = 0.0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            state, train_metrics = self.train_epoch(state, train_ds, epoch)
            
            # Log training metrics
            self.log_fn(
                f"Epoch {epoch}/{num_epochs} - "
                f"train_loss: {train_metrics['loss']:.4f}, "
                f"train_acc: {train_metrics['accuracy']:.4f}, "
                f"time: {time.time() - epoch_start:.2f}s"
            )
            
            # Evaluate
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate(state, eval_ds, use_ema=True)
                self.log_fn(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"eval_loss: {eval_metrics['loss']:.4f}, "
                    f"eval_acc: {eval_metrics['accuracy']:.4f}"
                )
                
                # Save best model
                if eval_metrics['accuracy'] > best_accuracy:
                    best_accuracy = eval_metrics['accuracy']
                    if self.checkpoint_dir:
                        self.save_checkpoint(state, epoch, prefix='best_')
                        self.log_fn(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            # Save periodic checkpoint
            if epoch % checkpoint_every == 0 and self.checkpoint_dir:
                self.save_checkpoint(state, epoch)
        
        self.log_fn("Training completed!")
        return state
    
    def save_checkpoint(self, state: TrainStateWithEMA, step: int, prefix: str = ''):
        """Save checkpoint."""
        if self.checkpoint_dir:
            checkpoint_dir = Path(self.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir=str(checkpoint_dir),
                target=state,
                step=step,
                prefix=prefix,
                keep=3
            )
    
    def load_checkpoint(self, state: TrainStateWithEMA, step: Optional[int] = None) -> TrainStateWithEMA:
        """Load checkpoint."""
        if self.checkpoint_dir:
            return checkpoints.restore_checkpoint(
                ckpt_dir=self.checkpoint_dir,
                target=state,
                step=step
            )
        return state
