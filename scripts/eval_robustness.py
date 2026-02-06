#!/usr/bin/env python3
"""Robustness evaluation script."""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_vision.data.loaders import ScalableDataLoader
from robust_vision.models.cnn import ProductionCNN
from robust_vision.training.state import TrainStateWithEMA
from robust_vision.evaluation.robustness import RobustnessEvaluator
from robust_vision.evaluation.visualization import create_robustness_report
from robust_vision.utils.config import load_config
from flax.training import checkpoints


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory or specific checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches to evaluate (for speed)'
    )
    
    parser.add_argument(
        '--use-ema',
        action='store_true',
        default=True,
        help='Use EMA parameters for evaluation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Initialize data loader
    print(f"Loading dataset: {config.training.dataset_name}")
    data_loader = ScalableDataLoader(
        dataset_name=config.training.dataset_name,
        batch_size=config.training.batch_size,
        image_size=tuple(config.training.image_size),
        cache=True,
        prefetch=True,
        augment=False  # No augmentation for evaluation
    )
    
    # Load test dataset
    test_ds = data_loader.get_test_loader()
    print("Test dataset loaded")
    
    # Initialize model
    print("Initializing model...")
    model = ProductionCNN(
        n_classes=config.model.n_classes,
        features=config.model.features,
        dropout_rate=config.model.dropout_rate,
        use_residual=config.model.use_residual
    )
    
    # Determine input shape
    input_shape = (1, config.training.image_size[0], config.training.image_size[1], 3)
    
    # Initialize parameters (needed for checkpoint loading)
    init_rng, rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.ones(input_shape), training=False)
    
    # Create dummy state for checkpoint loading
    import optax
    tx = optax.adam(learning_rate=config.training.learning_rate)
    state = TrainStateWithEMA.create_with_ema(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_decay=config.training.ema_decay
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    if checkpoint_path.is_dir():
        # Load latest checkpoint from directory
        state = checkpoints.restore_checkpoint(
            ckpt_dir=str(checkpoint_path),
            target=state
        )
    else:
        # Load specific checkpoint file
        state = checkpoints.restore_checkpoint(
            ckpt_dir=str(checkpoint_path.parent),
            target=state,
            step=int(checkpoint_path.stem.split('_')[-1])
        )
    
    print("Checkpoint loaded successfully")
    
    # Get parameters to use for evaluation
    if args.use_ema and state.ema_params is not None:
        print("Using EMA parameters for evaluation")
        eval_params = state.ema_params
    else:
        print("Using regular parameters for evaluation")
        eval_params = state.params
    
    # Initialize robustness evaluator
    print("Initializing robustness evaluator...")
    evaluator = RobustnessEvaluator(
        model_apply_fn=model.apply,
        params=eval_params,
        num_classes=config.model.n_classes,
        noise_types=['gaussian', 'salt_pepper', 'fog', 'occlusion'],
        severities=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        rng_key=rng
    )
    
    # Evaluate robustness
    print("\nEvaluating robustness...")
    results = evaluator.evaluate_dataset(
        dataset=test_ds,
        max_batches=args.max_batches
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    print(f"\nSaving results to {args.output}")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(results, args.output)
    
    # Create visualizations
    print("Creating visualizations...")
    create_robustness_report(results, args.output)
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
