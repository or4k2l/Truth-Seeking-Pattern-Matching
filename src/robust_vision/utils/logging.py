"""Logging utilities for training."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


def setup_logging(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging for training.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Optional experiment name (defaults to timestamp)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logger
    logger = logging.getLogger("robust_vision")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_path / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    log_dir: str = "./logs",
    experiment_name: str = "default"
):
    """
    Log metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step/epoch
        log_dir: Directory for log files
        experiment_name: Experiment name
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics file
    metrics_file = log_path / f"{experiment_name}_metrics.jsonl"
    
    # Prepare log entry
    log_entry = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        **metrics
    }
    
    # Append to file (JSONL format)
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


class MetricsLogger:
    """
    Metrics logger with support for TensorBoard-style logging.
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Optional experiment name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        
        # Initialize metrics file
        if self.metrics_file.exists():
            # Archive old file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = self.log_dir / f"{experiment_name}_metrics_{timestamp}.jsonl"
            self.metrics_file.rename(archive_file)
    
    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics for a given step.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch
        """
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **{k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def read_metrics(self) -> list:
        """
        Read all logged metrics.
        
        Returns:
            List of metric dictionaries
        """
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        return metrics


# Dummy WandB integration placeholder
class WandBLogger:
    """
    Placeholder for Weights & Biases integration.
    
    To use WandB:
    1. Install: pip install wandb
    2. Initialize: wandb.init(project="robust-vision")
    3. Log: wandb.log(metrics)
    """
    
    def __init__(self, project: str = "robust-vision", enabled: bool = False):
        self.enabled = enabled
        if enabled:
            try:
                import wandb
                self.wandb = wandb
                self.wandb.init(project=project)
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: int):
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def finish(self):
        if self.enabled:
            self.wandb.finish()
