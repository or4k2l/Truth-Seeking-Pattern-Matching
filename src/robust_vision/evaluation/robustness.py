"""Robustness evaluation utilities."""

from typing import Dict, List, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from ..data.noise import NoiseLibrary
from ..training.losses import compute_accuracy


class RobustnessEvaluator:
    """
    Evaluate model robustness across different noise types and severities.
    """
    
    def __init__(
        self,
        model_apply_fn,
        params,
        num_classes: int,
        noise_types: Optional[List[str]] = None,
        severities: Optional[List[float]] = None,
        rng_key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize robustness evaluator.
        
        Args:
            model_apply_fn: Model's apply function
            params: Model parameters
            num_classes: Number of output classes
            noise_types: List of noise types to evaluate
            severities: List of noise severities to test
            rng_key: JAX random key
        """
        self.model_apply_fn = model_apply_fn
        self.params = params
        self.num_classes = num_classes
        
        # Default noise types
        self.noise_types = noise_types or [
            "gaussian",
            "salt_pepper",
            "fog",
            "occlusion"
        ]
        
        # Default severities (0 = clean, higher = more severe)
        self.severities = severities or [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
        ]
        
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(42)
        self.noise_library = NoiseLibrary(rng_key=self.rng_key)
    
    def predict(self, images: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Get model predictions.
        
        Args:
            images: Input images
            training: Whether in training mode
            
        Returns:
            Logits
        """
        return self.model_apply_fn(self.params, images, training=training)
    
    def evaluate_clean(
        self,
        images: jnp.ndarray,
        labels: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate on clean images.
        
        Args:
            images: Clean images
            labels: Ground truth labels
            
        Returns:
            Dictionary with accuracy and confidence metrics
        """
        logits = self.predict(images)
        accuracy = compute_accuracy(logits, labels)
        
        # Compute confidence metrics
        probs = jax.nn.softmax(logits)
        max_probs = jnp.max(probs, axis=-1)
        mean_confidence = jnp.mean(max_probs)
        
        # Compute margins (difference between top-1 and top-2)
        sorted_logits = jnp.sort(logits, axis=-1)
        margins = sorted_logits[:, -1] - sorted_logits[:, -2]
        mean_margin = jnp.mean(margins)
        
        return {
            'accuracy': float(accuracy),
            'mean_confidence': float(mean_confidence),
            'mean_margin': float(mean_margin)
        }
    
    def evaluate_noisy(
        self,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        noise_type: str,
        severity: float
    ) -> Dict[str, float]:
        """
        Evaluate on noisy images.
        
        Args:
            images: Clean images
            labels: Ground truth labels
            noise_type: Type of noise
            severity: Noise severity
            
        Returns:
            Dictionary with accuracy and confidence metrics
        """
        # Generate noisy images
        self.rng_key, noise_key = jax.random.split(self.rng_key)
        noisy_images = self.noise_library.apply_noise(
            images, noise_type, severity, rng_key=noise_key
        )
        
        # Evaluate
        return self.evaluate_clean(noisy_images, labels)
    
    def evaluate_dataset(
        self,
        dataset,
        max_batches: Optional[int] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Evaluate robustness on entire dataset.
        
        Args:
            dataset: Dataset to evaluate on
            max_batches: Maximum number of batches to evaluate (for speed)
            
        Returns:
            Dictionary mapping noise types to list of results per severity
        """
        results = {noise_type: [] for noise_type in self.noise_types}
        results['clean'] = []
        
        print("Evaluating robustness across noise types and severities...")
        
        for severity in tqdm(self.severities, desc="Severities"):
            # Initialize metrics collectors for this severity
            severity_metrics = {
                noise_type: {'accuracy': [], 'mean_confidence': [], 'mean_margin': []}
                for noise_type in self.noise_types
            }
            severity_metrics['clean'] = {'accuracy': [], 'mean_confidence': [], 'mean_margin': []}
            
            # Evaluate across dataset
            batch_count = 0
            for batch in dataset:
                if max_batches and batch_count >= max_batches:
                    break
                
                # Convert to numpy if needed
                if hasattr(batch, 'numpy'):
                    images, labels = batch[0].numpy(), batch[1].numpy()
                else:
                    images, labels = batch
                
                # Convert to JAX arrays
                images = jnp.array(images)
                labels = jnp.array(labels)
                
                # Evaluate clean (only for severity 0)
                if severity == 0.0:
                    clean_metrics = self.evaluate_clean(images, labels)
                    for key in clean_metrics:
                        severity_metrics['clean'][key].append(clean_metrics[key])
                
                # Evaluate each noise type
                for noise_type in self.noise_types:
                    noisy_metrics = self.evaluate_noisy(images, labels, noise_type, severity)
                    for key in noisy_metrics:
                        severity_metrics[noise_type][key].append(noisy_metrics[key])
                
                batch_count += 1
            
            # Average metrics for this severity
            for noise_type in self.noise_types:
                avg_metrics = {
                    key: float(np.mean(severity_metrics[noise_type][key]))
                    for key in severity_metrics[noise_type]
                }
                avg_metrics['severity'] = severity
                results[noise_type].append(avg_metrics)
            
            if severity == 0.0:
                avg_clean_metrics = {
                    key: float(np.mean(severity_metrics['clean'][key]))
                    for key in severity_metrics['clean']
                }
                avg_clean_metrics['severity'] = severity
                results['clean'].append(avg_clean_metrics)
        
        return results
    
    def save_results(
        self,
        results: Dict[str, List[Dict[str, float]]],
        output_dir: str = "./results"
    ):
        """
        Save robustness results to CSV.
        
        Args:
            results: Results dictionary from evaluate_dataset
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save overall results
        all_results = []
        for noise_type, metrics_list in results.items():
            for metrics in metrics_list:
                all_results.append({
                    'noise_type': noise_type,
                    **metrics
                })
        
        df = pd.DataFrame(all_results)
        csv_path = output_path / "robustness_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save per-noise-type results
        for noise_type, metrics_list in results.items():
            df_noise = pd.DataFrame(metrics_list)
            csv_noise_path = output_path / f"robustness_{noise_type}.csv"
            df_noise.to_csv(csv_noise_path, index=False)
    
    def print_summary(self, results: Dict[str, List[Dict[str, float]]]):
        """
        Print summary of robustness results.
        
        Args:
            results: Results dictionary from evaluate_dataset
        """
        print("\n" + "="*60)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("="*60)
        
        for noise_type, metrics_list in results.items():
            print(f"\n{noise_type.upper()}:")
            print(f"  {'Severity':<12} {'Accuracy':<12} {'Confidence':<12} {'Margin':<12}")
            print(f"  {'-'*48}")
            for metrics in metrics_list:
                print(
                    f"  {metrics['severity']:<12.2f} "
                    f"{metrics['accuracy']:<12.4f} "
                    f"{metrics['mean_confidence']:<12.4f} "
                    f"{metrics['mean_margin']:<12.4f}"
                )
        
        print("="*60)
