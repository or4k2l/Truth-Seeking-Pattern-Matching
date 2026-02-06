#!/usr/bin/env python3
"""Main training script for robust vision models."""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_vision.data.loaders import ScalableDataLoader
from robust_vision.models.cnn import ProductionCNN
from robust_vision.training.trainer import ProductionTrainer
from robust_vision.utils.config import load_config, validate_config
from robust_vision.utils.logging import setup_logging, MetricsLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robust Vision Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/train.py --config configs/baseline.yaml
  python scripts/train.py --config configs/margin_loss.yaml --seed 1234
  
For more info: https://github.com/or4k2l/robust-vision
'''
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name for logging'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load and validate config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Setup logging
    logger = setup_logging(
        log_dir=config.training.log_dir,
        experiment_name=args.experiment_name
    )
    logger.info("Starting training...")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(
        log_dir=config.training.log_dir,
        experiment_name=args.experiment_name
    )
    metrics_logger.log_config(config.to_dict())
    
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Initialize data loader
    logger.info(f"Loading dataset: {config.training.dataset_name}")
    data_loader = ScalableDataLoader(
        dataset_name=config.training.dataset_name,
        batch_size=config.training.batch_size,
        image_size=tuple(config.training.image_size),
        cache=True,
        prefetch=True,
        augment=True
    )
    
    # Load datasets (TF datasets, not iterators, so they can be reused)
    train_ds = data_loader.create_dataset('train', repeat=False)
    eval_ds = data_loader.create_dataset('test', repeat=False)
    
    logger.info(f"Training dataset loaded")
    logger.info(f"Evaluation dataset loaded")
    
    # Initialize model
    logger.info("Initializing model...")
    model = ProductionCNN(
        n_classes=config.model.n_classes,
        features=config.model.features,
        dropout_rate=config.model.dropout_rate,
        use_residual=config.model.use_residual
    )
    
    # Prepare loss kwargs
    loss_kwargs = {
        'smoothing': config.training.label_smoothing,
        'margin': config.training.margin,
        'margin_weight': config.training.margin_weight
    }
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = ProductionTrainer(
        model=model,
        num_classes=config.model.n_classes,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        ema_decay=config.training.ema_decay if config.training.ema_enabled else 0.0,
        loss_type=config.training.loss_type,
        loss_kwargs=loss_kwargs,
        checkpoint_dir=config.training.checkpoint_dir,
        log_fn=logger.info
    )
    
    # Determine input shape
    input_shape = (1, config.training.image_size[0], config.training.image_size[1], 3)
    
    # Train
    logger.info(f"Starting training for {config.training.epochs} epochs...")
    final_state = trainer.train(
        rng=rng,
        train_ds=train_ds,
        eval_ds=eval_ds,
        num_epochs=config.training.epochs,
        input_shape=input_shape,
        eval_every=config.training.eval_every,
        checkpoint_every=config.training.checkpoint_every
    )
    
    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    trainer.save_checkpoint(final_state, config.training.epochs, prefix='final_')
    
    logger.info("Training completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
