"""Loss functions for robust training."""

import jax
import jax.numpy as jnp
import optax


def label_smoothing_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    smoothing: float = 0.1
) -> jnp.ndarray:
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing prevents overconfident predictions and improves generalization.
    
    Args:
        logits: Model predictions [batch, num_classes]
        labels: Ground truth labels [batch]
        num_classes: Number of classes
        smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform)
        
    Returns:
        Scalar loss value
    """
    # One-hot encode labels
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    
    # Apply label smoothing
    smooth_labels = one_hot_labels * (1 - smoothing) + smoothing / num_classes
    
    # Compute cross-entropy
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(smooth_labels * log_probs, axis=-1)
    
    return jnp.mean(loss)


def margin_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    margin: float = 1.0
) -> jnp.ndarray:
    """
    Margin loss for encouraging confident predictions.
    
    Encourages the correct class logit to be higher than other class logits
    by at least the specified margin.
    
    Args:
        logits: Model predictions [batch, num_classes]
        labels: Ground truth labels [batch]
        margin: Desired margin between correct and incorrect classes
        
    Returns:
        Scalar loss value
    """
    batch_size, num_classes = logits.shape
    
    # Get correct class logits
    correct_logits = jnp.take_along_axis(
        logits, 
        labels[:, None], 
        axis=1
    ).squeeze(1)
    
    # Get max incorrect class logits
    # Set correct class logits to -inf to exclude them
    mask = jax.nn.one_hot(labels, num_classes)
    masked_logits = jnp.where(mask, -jnp.inf, logits)
    max_incorrect_logits = jnp.max(masked_logits, axis=1)
    
    # Margin loss: max(0, margin - (correct - max_incorrect))
    loss = jnp.maximum(0, margin - (correct_logits - max_incorrect_logits))
    
    return jnp.mean(loss)


def focal_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> jnp.ndarray:
    """
    Focal loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples.
    
    Args:
        logits: Model predictions [batch, num_classes]
        labels: Ground truth labels [batch]
        num_classes: Number of classes
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        
    Returns:
        Scalar loss value
    """
    # Get probabilities
    probs = jax.nn.softmax(logits)
    
    # One-hot encode labels
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    
    # Get probability of correct class
    p_t = jnp.sum(probs * one_hot_labels, axis=-1)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Cross-entropy
    ce_loss = -jnp.log(p_t + 1e-8)
    
    # Focal loss
    loss = alpha * focal_weight * ce_loss
    
    return jnp.mean(loss)


def combined_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    loss_type: str = "label_smoothing",
    **loss_kwargs
) -> jnp.ndarray:
    """
    Combined loss function that can switch between different loss types.
    
    Args:
        logits: Model predictions
        labels: Ground truth labels
        num_classes: Number of classes
        loss_type: Type of loss ('label_smoothing', 'margin', 'focal', or 'combined')
        **loss_kwargs: Additional loss-specific arguments
        
    Returns:
        Scalar loss value
    """
    if loss_type == "label_smoothing":
        smoothing = loss_kwargs.get("smoothing", 0.1)
        return label_smoothing_cross_entropy(logits, labels, num_classes, smoothing)
    
    elif loss_type == "margin":
        margin = loss_kwargs.get("margin", 1.0)
        return margin_loss(logits, labels, margin)
    
    elif loss_type == "focal":
        alpha = loss_kwargs.get("alpha", 0.25)
        gamma = loss_kwargs.get("gamma", 2.0)
        return focal_loss(logits, labels, num_classes, alpha, gamma)
    
    elif loss_type == "combined":
        # Combine label smoothing cross-entropy with margin loss
        smoothing = loss_kwargs.get("smoothing", 0.1)
        margin = loss_kwargs.get("margin", 1.0)
        margin_weight = loss_kwargs.get("margin_weight", 0.5)
        
        ce_loss = label_smoothing_cross_entropy(logits, labels, num_classes, smoothing)
        m_loss = margin_loss(logits, labels, margin)
        
        return ce_loss + margin_weight * m_loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model predictions [batch, num_classes]
        labels: Ground truth labels [batch]
        
    Returns:
        Accuracy as a float
    """
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)
