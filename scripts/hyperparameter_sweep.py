#!/usr/bin/env python3
"""Hyperparameter sweep script."""

import argparse
import jax
from pathlib import Path
import sys
import itertools
from typing import Dict, List, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_vision.data.loaders import ScalableDataLoader
from robust_vision.models.cnn import ProductionCNN
from robust_vision.training.trainer import ProductionTrainer
from robust_vision.utils.config import Config, ModelConfig, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for robust vision models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./sweep_results',
        help='Output directory for sweep results'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs per configuration'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed'
    )
    
    return parser.parse_args()


def create_sweep_configs() -> List[Dict[str, Any]]:
    """
    Create configurations for hyperparameter sweep.
    
    Returns:
        List of configuration dictionaries
    """
    # Define hyperparameter grid
    learning_rates = [1e-4, 1e-3, 5e-3]
    dropout_rates = [0.1, 0.3, 0.5]
    loss_types = ['label_smoothing', 'margin', 'combined']
    label_smoothings = [0.0, 0.1, 0.2]
    
    configs = []
    config_id = 0
    
    for lr, dropout, loss_type, smoothing in itertools.product(
        learning_rates, dropout_rates, loss_types, label_smoothings
    ):
        config = {
            'id': config_id,
            'learning_rate': lr,
            'dropout_rate': dropout,
            'loss_type': loss_type,
            'label_smoothing': smoothing
        }
        configs.append(config)
        config_id += 1
    
    return configs


def run_single_experiment(
    config_dict: Dict[str, Any],
    args,
    rng: jax.random.PRNGKey
) -> Dict[str, Any]:
    """
    Run a single experiment with given configuration.
    
    Args:
        config_dict: Configuration dictionary
        args: Command line arguments
        rng: JAX random key
        
    Returns:
        Results dictionary
    """
    config_id = config_dict['id']
    print(f"\n{'='*60}")
    print(f"Running experiment {config_id}")
    print(f"Config: {config_dict}")
    print(f"{'='*60}\n")
    
    # Create configuration
    model_config = ModelConfig(
        n_classes=10,
        features=[64, 128, 256],
        dropout_rate=config_dict['dropout_rate'],
        use_residual=True
    )
    
    training_config = TrainingConfig(
        batch_size=128,
        epochs=args.epochs,
        learning_rate=config_dict['learning_rate'],
        weight_decay=1e-4,
        loss_type=config_dict['loss_type'],
        label_smoothing=config_dict['label_smoothing'],
        margin=1.0,
        margin_weight=0.5,
        ema_enabled=True,
        ema_decay=0.99,
        eval_every=1,
        checkpoint_every=args.epochs,  # Only save at end
        dataset_name=args.dataset,
        image_size=[32, 32],
        checkpoint_dir=f"{args.output}/exp_{config_id}/checkpoints",
        log_dir=f"{args.output}/exp_{config_id}/logs"
    )
    
    config = Config(model=model_config, training=training_config)
    
    # Initialize data loader
    data_loader = ScalableDataLoader(
        dataset_name=args.dataset,
        batch_size=128,
        image_size=(32, 32),
        cache=True,
        prefetch=True,
        augment=True
    )
    
    train_ds = data_loader.get_train_loader()
    eval_ds = data_loader.get_test_loader()
    
    # Initialize model
    model = ProductionCNN(
        n_classes=model_config.n_classes,
        features=model_config.features,
        dropout_rate=model_config.dropout_rate,
        use_residual=model_config.use_residual
    )
    
    # Prepare loss kwargs
    loss_kwargs = {
        'smoothing': training_config.label_smoothing,
        'margin': training_config.margin,
        'margin_weight': training_config.margin_weight
    }
    
    # Initialize trainer
    trainer = ProductionTrainer(
        model=model,
        num_classes=model_config.n_classes,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        ema_decay=training_config.ema_decay,
        loss_type=training_config.loss_type,
        loss_kwargs=loss_kwargs,
        checkpoint_dir=training_config.checkpoint_dir,
        log_fn=print
    )
    
    # Train
    input_shape = (1, 32, 32, 3)
    final_state = trainer.train(
        rng=rng,
        train_ds=train_ds,
        eval_ds=eval_ds,
        num_epochs=args.epochs,
        input_shape=input_shape,
        eval_every=1,
        checkpoint_every=args.epochs
    )
    
    # Evaluate final performance
    final_metrics = trainer.evaluate(final_state, eval_ds, use_ema=True)
    
    # Prepare results
    results = {
        'config_id': config_id,
        'config': config_dict,
        'final_accuracy': float(final_metrics['accuracy']),
        'final_loss': float(final_metrics['loss'])
    }
    
    print(f"\nExperiment {config_id} completed!")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    print(f"Final loss: {results['final_loss']:.4f}")
    
    return results


def main():
    """Main sweep function."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create sweep configurations
    print("Creating hyperparameter sweep configurations...")
    sweep_configs = create_sweep_configs()
    print(f"Total configurations to evaluate: {len(sweep_configs)}")
    
    # Save sweep configurations
    with open(output_path / "sweep_configs.json", 'w') as f:
        json.dump(sweep_configs, f, indent=2)
    
    # Run experiments
    all_results = []
    rng = jax.random.PRNGKey(args.seed)
    
    for i, config_dict in enumerate(sweep_configs):
        # Split RNG for each experiment
        rng, exp_rng = jax.random.split(rng)
        
        try:
            results = run_single_experiment(config_dict, args, exp_rng)
            all_results.append(results)
            
            # Save intermediate results
            with open(output_path / "sweep_results.json", 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"Error in experiment {config_dict['id']}: {e}")
            continue
    
    # Find best configuration
    if all_results:
        best_result = max(all_results, key=lambda x: x['final_accuracy'])
        
        print("\n" + "="*60)
        print("SWEEP COMPLETED!")
        print("="*60)
        print(f"Total experiments: {len(all_results)}")
        print(f"\nBest configuration (ID {best_result['config_id']}):")
        print(f"  Config: {best_result['config']}")
        print(f"  Accuracy: {best_result['final_accuracy']:.4f}")
        print(f"  Loss: {best_result['final_loss']:.4f}")
        print(f"\nAll results saved to {args.output}")
        
        # Save best config
        with open(output_path / "best_config.json", 'w') as f:
            json.dump(best_result, f, indent=2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
