# Reproducing Research Results

This guide shows how to reproduce key findings from our paper:  
**"A Systematic Decomposition of Neural Network Robustness"**

---

## Overview

Our research identified three key factors affecting neural network robustness:

1. **Loss Functions** (375× impact)
2. **Learning Rules** (133× impact)  
3. **Hardware Constraints** (-62% penalty)

This framework implements the production-ready versions of these findings.

---

## Quick Reproduction

### Experiment 1: Margin Loss vs Cross-Entropy

**Research Finding:** Margin loss achieves 375× higher SNR than standard cross-entropy.

```bash
# Train with margin loss
robust-vision-train --config configs/research/margin_ablation.yaml

# Train with standard loss (baseline)
robust-vision-train --config configs/research/baseline_comparison.yaml

# Compare results
python scripts/compare_experiments.py \
  --exp1 ./checkpoints/research/margin_lambda_10 \
  --exp2 ./checkpoints/research/baseline_ce \
  --output ./comparison_results
```

**Expected Results:**

| Method | SNR | Accuracy |
|--------|-----|----------|
| Cross-Entropy | ~6-10 | 98% |
| Margin (λ=10) | ~2000+ | 98% |
| **Improvement** | **200-375×** | Same |

---

### Experiment 2: Lambda Ablation Study

**Research Finding:** Margin loss performance scales with λ parameter.

```bash
# Run hyperparameter sweep
python scripts/hyperparameter_sweep.py \
  --config configs/research/lambda_sweep.yaml \
  --output ./sweep_results
```

**Expected Trend:**

```
λ = 0.1  → SNR ~15   (weak margin)
λ = 1.0  → SNR ~75   (moderate margin)
λ = 10.0 → SNR ~2400 (strong margin)
λ = 20.0 → SNR ~2300 (diminishing returns)
```

---

### Experiment 3: Robustness Evaluation

**Research Finding:** High SNR correlates with robustness under noise.

```bash
# Evaluate model on multiple noise types
robust-vision-eval \
  --checkpoint ./checkpoints/research/margin_lambda_10/best \
  --config configs/research/margin_ablation.yaml \
  --output ./robustness_results
```

**Expected Robustness Curves:**

At 50% Gaussian Noise:
- Standard model: Accuracy drops to ~60%
- Margin model (λ=10): Accuracy maintains ~95% ✅

---

## Detailed Reproduction

### Setup

```bash
# Clone the repository
git clone https://github.com/or4k2l/robust-vision.git
cd robust-vision

# Install dependencies
pip install -e .

# Create research output directories
mkdir -p results/research
mkdir -p checkpoints/research
```

---

### Full Experimental Pipeline

#### Step 1: Train All Variants

```bash
# Baseline (Standard Cross-Entropy)
robust-vision-train --config configs/research/baseline_comparison.yaml

# Margin Loss λ=1
python scripts/train.py --config configs/research/margin_ablation.yaml \
  --override training.margin_lambda=1.0 \
  --override training.checkpoint_dir=./checkpoints/research/margin_lambda_1

# Margin Loss λ=10 (Best)
robust-vision-train --config configs/research/margin_ablation.yaml

# Margin Loss λ=20
python scripts/train.py --config configs/research/margin_ablation.yaml \
  --override training.margin_lambda=20.0 \
  --override training.checkpoint_dir=./checkpoints/research/margin_lambda_20
```

#### Step 2: Evaluate All Models

```bash
for lambda in baseline 1 10 20; do
  robust-vision-eval \
    --checkpoint ./checkpoints/research/margin_lambda_${lambda}/best \
    --config configs/research/margin_ablation.yaml \
    --output ./results/research/eval_lambda_${lambda}
done
```

#### Step 3: Generate Comparison Plots

```bash
python scripts/research/plot_ablation_results.py \
  --results_dir ./results/research \
  --output ./paper_figures \
  --style publication \
  --dpi 300
```

---

## Expected Outputs

### Training Logs

```
Epoch 1/30
  Train Loss: 0.5234  Train Acc: 0.8123  SNR: 45.2
  Val Loss:   0.4821  Val Acc:   0.8345  SNR: 52.1

Epoch 15/30
  Train Loss: 0.1234  Train Acc: 0.9678  SNR: 1834.2
  Val Loss:   0.1456  Val Acc:   0.9612  SNR: 1456.7
  
Epoch 30/30 ✓
  Train Loss: 0.0523  Train Acc: 0.9845  SNR: 2398.1
  Val Loss:   0.0687  Val Acc:   0.9789  SNR: 2124.5
  
✓ Best checkpoint saved: epoch 28, SNR=2456.3
```

### Robustness Evaluation Summary

```
ROBUSTNESS EVALUATION RESULTS
════════════════════════════════════════

Model: margin_lambda_10

GAUSSIAN NOISE:
  Level    Accuracy    SNR      Degradation
  ───────────────────────────────────────
  0.0      0.9789     2124.5   —
  0.1      0.9623     1856.2   -1.7%
  0.2      0.9412     1523.8   -3.8%
  0.3      0.9178     1245.6   -6.2%
  0.5      0.8534      892.3   -12.8%
  0.7      0.7823      534.7   -20.1%

SALT & PEPPER:
  Level    Accuracy    SNR      Degradation
  ───────────────────────────────────────
  0.0      0.9789     2124.5   —
  0.1      0.9645     1923.4   -1.5%
  ...

COMPARISON WITH BASELINE:
  At 50% Gaussian noise:
    Baseline:  Accuracy = 0.6234  SNR = 4.2
    Margin:    Accuracy = 0.8534  SNR = 892.3
    
    Improvement: +36.9% accuracy, +212× SNR
```

---

## Visualizations

The framework automatically generates:

### 1. Training Curves
- Loss vs Epoch
- Accuracy vs Epoch  
- **SNR vs Epoch** (unique to this framework)

### 2. Robustness Curves
- Accuracy vs Noise Level (for each noise type)
- SNR vs Noise Level
- Degradation curves

### 3. Comparison Plots
- Side-by-side model comparisons
- Lambda ablation results
- Confidence distribution histograms

Example output:
```
./paper_figures/
├── training_curves_margin.pdf
├── robustness_curves_comparison.pdf
├── lambda_ablation_snr.pdf
└── confidence_distributions.pdf
```

---

## Validation Checklist

To verify successful reproduction:

- [ ] Margin model achieves SNR > 2000 on clean data
- [ ] Baseline model achieves SNR < 20 on clean data
- [ ] Margin model maintains >85% accuracy at 50% Gaussian noise
- [ ] Baseline model drops to <70% accuracy at 50% Gaussian noise
- [ ] SNR scales roughly linearly with lambda (up to λ=10)
- [ ] All plots generated successfully in publication quality

---

## Troubleshooting

### Issue: SNR values too low

**Possible causes:**
1. Learning rate too high (causes instability)
2. Margin lambda too low (weak margin enforcement)
3. Not enough training epochs

**Solution:**
```yaml
training:
  learning_rate: 0.0005  # Reduce from 0.001
  margin_lambda: 10.0    # Ensure this is set
  epochs: 40             # Increase if needed
```

### Issue: Training diverges

**Possible causes:**
1. Lambda too high
2. Learning rate too high

**Solution:**
```yaml
training:
  margin_lambda: 5.0   # Reduce from 10.0
  learning_rate: 0.0001
```

### Issue: Out of memory

**Solution:**
```yaml
training:
  batch_size: 64  # Reduce from 128
```

---

## Citation

If you use these experimental configurations, please cite both:

1. **The framework:**
```bibtex
@software{robust_vision_2026,
  author = {Akbay, Yahya},
  title = {Robust Vision: Production-Ready Scalable Training Framework},
  year = {2026},
  url = {https://github.com/or4k2l/robust-vision}
}
```

2. **The research paper:**
```bibtex
@article{akbay2025robustness,
  title={A Systematic Decomposition of Neural Network Robustness},
  author={Akbay, Yahya},
  journal={arXiv preprint arXiv:2502.XXXXX},
  year={2025}
}
```

---

## Questions?

- **Framework issues:** [Open an issue](https://github.com/or4k2l/robust-vision/issues)
- **Research questions:** oneochrone@gmail.com
- **Paper discussion:** [arXiv comments](https://arxiv.org)

---

**Last Updated:** February 2026
