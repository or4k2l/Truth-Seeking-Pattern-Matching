# Robust Vision: Production-Ready Scalable Training Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, scalable framework for training robust vision models with advanced techniques including EMA, label smoothing, margin loss, and multi-GPU support.

## 🎯 Features

- **Production-Ready Code**: Clean, maintainable, tested codebase
- **Scalable Training**: Single GPU → Multi-GPU with zero code changes
- **Advanced Techniques**: 
  - Exponential Moving Average (EMA) for stable predictions
  - Label smoothing for better generalization
  - Margin loss for confident predictions
  - Mixup augmentation
- **Comprehensive Robustness Evaluation**: Test against 4 noise types
- **Easy to Use**: Train a model in 3 commands
- **Full Documentation**: Installation, training, and deployment guides

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/or4k2l/robust-vision.git
cd robust-vision
pip install -r requirements.txt
pip install -e .
```

### Train a Model

```bash
python scripts/train.py --config configs/baseline.yaml
```

### Evaluate Robustness

```bash
python scripts/eval_robustness.py \
  --checkpoint ./checkpoints/baseline/best_checkpoint_18 \
  --config configs/baseline.yaml \
  --output ./results
```

That's it! You now have a trained model and robustness evaluation results.

## 📊 What This Framework Does

This framework trains vision models that are robust to real-world noise and perturbations. It evaluates models across multiple noise types:

- **Gaussian Noise**: Random pixel-level noise
- **Salt & Pepper**: Random black/white pixels
- **Fog**: Atmospheric haze effects
- **Occlusion**: Random patches blocking view

The framework automatically generates robustness curves showing how accuracy degrades under increasing noise levels.

## 🎨 Example Results

Train a model and get automatic robustness curves:

```
ROBUSTNESS EVALUATION SUMMARY
============================================================

GAUSSIAN:
  Severity     Accuracy     Confidence   Margin      
  ------------------------------------------------
  0.00         0.9850       0.9820       2.3400      
  0.10         0.9420       0.9350       1.8900      
  0.20         0.8850       0.8720       1.4200      
  0.30         0.8120       0.7980       0.9800      

SALT_PEPPER:
  Severity     Accuracy     Confidence   Margin      
  ------------------------------------------------
  0.00         0.9850       0.9820       2.3400      
  0.10         0.9580       0.9490       2.0100      
  0.20         0.9210       0.9080       1.6500      
...
```

## 📦 Repository Structure

```
.
├── src/robust_vision/          # Main package
│   ├── data/                   # Data loading and noise
│   ├── models/                 # Model architectures
│   ├── training/               # Training logic
│   ├── evaluation/             # Robustness evaluation
│   └── utils/                  # Config and logging
├── scripts/                    # Training/evaluation scripts
│   ├── train.py
│   ├── eval_robustness.py
│   └── hyperparameter_sweep.py
├── configs/                    # Configuration files
│   ├── baseline.yaml
│   └── margin_loss.yaml
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── notebooks/                  # Example notebooks
└── requirements.txt
```

## 🛠️ Configuration

Create custom training configurations in YAML:

```yaml
model:
  n_classes: 10
  features: [64, 128, 256]
  dropout_rate: 0.3

training:
  batch_size: 128
  epochs: 30
  learning_rate: 0.001
  loss_type: "combined"  # label_smoothing, margin, focal, combined
  
  # EMA for stable predictions
  ema_enabled: true
  ema_decay: 0.99
  
  dataset_name: "cifar10"
```

## 🎓 Key Techniques

### 1. Exponential Moving Average (EMA)

EMA tracks a moving average of model parameters during training, providing more stable and often better predictions:

```python
# Automatically handled by the framework
ema_params = decay * ema_params + (1 - decay) * params
```

### 2. Label Smoothing

Prevents overconfident predictions by smoothing target distributions:

```python
smooth_labels = one_hot * (1 - smoothing) + smoothing / num_classes
```

### 3. Margin Loss

Encourages larger separation between correct and incorrect classes:

```python
loss = max(0, margin - (correct_logit - max_incorrect_logit))
```

### 4. Multi-GPU Training

Automatic parallelization across GPUs with JAX's `pmap`:

```bash
# Uses all available GPUs automatically
python scripts/train.py --config configs/baseline.yaml
```

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Training Guide](docs/TRAINING.md)**: How to train models
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Production deployment

## 🧪 Testing

Run tests to verify your installation:

```bash
pip install pytest
pytest tests/
```

## 🐳 Docker

Build and run with Docker:

```bash
docker build -t robust-vision:latest .
docker run --gpus all robust-vision:latest
```

## 📈 Hyperparameter Tuning

Automated hyperparameter search:

```bash
python scripts/hyperparameter_sweep.py \
  --output ./sweep_results \
  --epochs 10
```

## 🔍 Use Cases

This framework is ideal for:

- **Autonomous Driving**: Train robust perception models
- **Medical Imaging**: Handle noisy/corrupted medical scans
- **Robotics**: Vision systems robust to environmental variations
- **Security**: Models resistant to adversarial perturbations
- **Research**: Benchmark robustness of new architectures

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{robust_vision_2026,
  author = {Akbay, Yahya},
  title = {Robust Vision: Production-Ready Scalable Training Framework},
  year = {2026},
  url = {https://github.com/or4k2l/robust-vision}
}
```

See [CITATION.cff](CITATION.cff) for more details.

## 🙏 Acknowledgments

Built with:
- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [Flax](https://github.com/google/flax) - Neural network library
- [Optax](https://github.com/deepmind/optax) - Gradient processing
- [TensorFlow Datasets](https://www.tensorflow.org/datasets) - Dataset loading

## 📧 Contact

For questions or issues, please [open an issue](https://github.com/or4k2l/robust-vision/issues) on GitHub.

---

**⭐ Star this repo if you find it useful!**
