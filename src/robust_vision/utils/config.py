"""Configuration management."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    n_classes: int = 10
    features: List[int] = field(default_factory=lambda: [64, 128, 256])
    dropout_rate: float = 0.3
    use_residual: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss configuration
    loss_type: str = "label_smoothing"
    label_smoothing: float = 0.1
    margin: float = 1.0
    margin_weight: float = 0.5
    
    # Augmentation
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # EMA
    ema_enabled: bool = True
    ema_decay: float = 0.99
    
    # Evaluation
    eval_every: int = 1
    checkpoint_every: int = 5
    
    # Data
    dataset_name: str = "cifar10"
    image_size: List[int] = field(default_factory=lambda: [32, 32])
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        model = ModelConfig(**config_dict.get('model', {}))
        training = TrainingConfig(**config_dict.get('training', {}))
        return cls(model=model, training=training)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        config_path: Path to save YAML config file
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def validate_config(config: Config) -> bool:
    """
    Validate configuration.
    
    Args:
        config: Config object
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Validate model config
    if config.model.n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    
    if config.model.dropout_rate < 0 or config.model.dropout_rate >= 1:
        raise ValueError("dropout_rate must be in [0, 1)")
    
    # Validate training config
    if config.training.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    
    if config.training.epochs < 1:
        raise ValueError("epochs must be >= 1")
    
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    
    if config.training.label_smoothing < 0 or config.training.label_smoothing >= 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    
    if config.training.ema_decay < 0 or config.training.ema_decay >= 1:
        raise ValueError("ema_decay must be in [0, 1)")
    
    return True
