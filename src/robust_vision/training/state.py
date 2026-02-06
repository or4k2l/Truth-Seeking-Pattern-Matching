"""Training state with EMA parameter tracking."""

from typing import Any, Optional
from flax.training import train_state
import optax
import jax.numpy as jnp
from flax import struct


class TrainStateWithEMA(train_state.TrainState):
    """
    Training state with Exponential Moving Average (EMA) of parameters.
    
    EMA parameters provide more stable predictions and often better generalization.
    """
    
    ema_params: Optional[Any] = None
    ema_decay: float = 0.99
    
    def apply_ema_update(self):
        """Update EMA parameters using current parameters."""
        if self.ema_params is None:
            # Initialize EMA params with current params
            return self.replace(ema_params=self.params)
        else:
            # Update EMA: ema = decay * ema + (1 - decay) * params
            new_ema = jax.tree_map(
                lambda ema, p: self.ema_decay * ema + (1 - self.ema_decay) * p,
                self.ema_params,
                self.params
            )
            return self.replace(ema_params=new_ema)
    
    @classmethod
    def create_with_ema(
        cls,
        *,
        apply_fn,
        params,
        tx,
        ema_decay: float = 0.99,
        **kwargs
    ):
        """
        Create training state with EMA.
        
        Args:
            apply_fn: Model's apply function
            params: Initial parameters
            tx: Optax optimizer
            ema_decay: EMA decay rate
            **kwargs: Additional arguments for TrainState
            
        Returns:
            TrainStateWithEMA instance
        """
        state = cls.create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            **kwargs
        )
        # Initialize EMA params with current params
        return state.replace(
            ema_params=params,
            ema_decay=ema_decay
        )


def create_train_state(
    rng,
    model,
    learning_rate: float,
    weight_decay: float = 0.0,
    ema_decay: float = 0.99,
    input_shape: tuple = (1, 32, 32, 3)
) -> TrainStateWithEMA:
    """
    Create a training state with optimizer and EMA.
    
    Args:
        rng: JAX random key
        model: Flax model
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        ema_decay: EMA decay rate
        input_shape: Shape of input for initialization
        
    Returns:
        TrainStateWithEMA instance
    """
    # Initialize parameters
    params = model.init(rng, jnp.ones(input_shape), training=False)
    
    # Create optimizer with weight decay
    if weight_decay > 0:
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        tx = optax.adam(learning_rate=learning_rate)
    
    # Create training state with EMA
    state = TrainStateWithEMA.create_with_ema(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_decay=ema_decay
    )
    
    return state
