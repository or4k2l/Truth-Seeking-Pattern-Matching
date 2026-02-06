"""Production-ready CNN model with modern best practices."""

from typing import Optional, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    features: int
    kernel_size: Sequence[int] = (3, 3)
    strides: Sequence[int] = (1, 1)
    dropout_rate: float = 0.0
    training: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        # First conv
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Dropout
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Second conv
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Adjust residual if needed (for stride or channel mismatch)
        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                padding='SAME'
            )(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)
        
        # Add residual and activate
        x = nn.relu(x + residual)
        
        return x


class ProductionCNN(nn.Module):
    """
    Production-ready CNN with modern best practices.
    
    Features:
    - Batch normalization
    - Dropout regularization
    - Residual connections
    - Configurable architecture
    """
    
    n_classes: int = 10
    features: Sequence[int] = (64, 128, 256)
    dropout_rate: float = 0.3
    use_residual: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Forward pass.
        
        Args:
            x: Input images [batch, height, width, channels]
            training: Whether in training mode (affects BatchNorm and Dropout)
            
        Returns:
            Logits [batch, n_classes]
        """
        # Initial conv layer
        x = nn.Conv(
            features=self.features[0],
            kernel_size=(3, 3),
            padding='SAME'
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Build residual/conv blocks
        for i, num_features in enumerate(self.features):
            if self.use_residual:
                # Residual block
                stride = (2, 2) if i > 0 else (1, 1)
                x = ResidualBlock(
                    features=num_features,
                    strides=stride,
                    dropout_rate=self.dropout_rate
                )(x, training=training)
            else:
                # Regular conv block
                if i > 0:
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                
                x = nn.Conv(
                    features=num_features,
                    kernel_size=(3, 3),
                    padding='SAME'
                )(x)
                x = nn.BatchNorm(use_running_average=not training)(x)
                x = nn.relu(x)
                
                if self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # Dropout before final layer
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Final classification layer
        x = nn.Dense(features=self.n_classes)(x)
        
        return x
    
    def get_features(self, x, training: bool = False):
        """
        Extract features before final classification layer.
        
        Args:
            x: Input images
            training: Whether in training mode
            
        Returns:
            Feature vectors before final dense layer
        """
        # Run through all layers except final dense
        for i, num_features in enumerate(self.features):
            if i == 0:
                x = nn.Conv(features=num_features, kernel_size=(3, 3), padding='SAME')(x)
                x = nn.BatchNorm(use_running_average=not training)(x)
                x = nn.relu(x)
            else:
                if self.use_residual:
                    x = ResidualBlock(
                        features=num_features,
                        strides=(2, 2),
                        dropout_rate=self.dropout_rate
                    )(x, training=training)
                else:
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
                    x = nn.Conv(features=num_features, kernel_size=(3, 3), padding='SAME')(x)
                    x = nn.BatchNorm(use_running_average=not training)(x)
                    x = nn.relu(x)
        
        x = jnp.mean(x, axis=(1, 2))
        return x
