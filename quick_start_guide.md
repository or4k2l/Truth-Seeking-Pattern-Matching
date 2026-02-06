# Quick Start Guide

Get up and running in 5 minutes.

## Installation

### Option 1: Standard Installation (CPU)

```bash
git clone https://github.com/or4k2l/truth-seeking-pattern-matching.git
cd truth-seeking-pattern-matching
pip install -r requirements.txt
```

### Option 2: GPU Support (Faster)

```bash
git clone https://github.com/or4k2l/truth-seeking-pattern-matching.git
cd truth-seeking-pattern-matching

# For CUDA 11
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Then install other requirements
pip install pandas matplotlib seaborn tqdm pillow kagglehub
```

### Option 3: Google Colab (No Installation)

[Open In Colab](https://colab.research.google.com/github/or4k2l/truth-seeking-pattern-matching/blob/main/notebook.ipynb)

Just click the badge and run.

---

## Running Your First Experiment

### Basic Benchmark (5 minutes)

```bash
python physically_inspired_pattern_matching.py
```

This will:
1. Download KITTI dataset (happens once)
2. Run 210 tests
3. Generate 3 visualization files in `outputs/`

Expected output:
```
Physical Crossbar:  Mean SNR: 181.89  |  Accuracy: 100.0%
Digital Baseline:   Mean SNR: 1.75    |  Accuracy: 30.0%
```

### Extended Benchmark (20 minutes)

```bash
python extended_benchmark.py
```

This runs the full analysis:
- Ablation study (5 conductance ranges)
- 100 images for statistics
- CNN comparison
- Theory visualizations

---

## Understanding the Results

### Key Metrics

**SNR (Signal-to-Noise Ratio):**
- Measures how confident the model is
- `SNR = output_target / output_control`
- Higher = more confident

**Accuracy:**
- Did it classify correctly (Yes or No)
- Both can be 100% but with different confidence

### Example Interpretation

```
Model A: SNR = 200, Accuracy = 100%
Model B: SNR = 1.5, Accuracy = 100%
```

Both are correct, but:
- Model A is ultra confident (200:1 ratio)
- Model B is barely sure (1.5:1 ratio)

In safety-critical systems, Model A is better.

---

## Exploring the Code

### Minimal Example

```python
import jax.numpy as jnp
from physically_inspired_pattern_matching import PhysicalCrossbar, ModelConfig

# Setup
config = ModelConfig(n_inputs=4096, n_outputs=2)
model = PhysicalCrossbar(config, g_min=0.0, g_max=1.0)

# Train on your image
image = jnp.array([...])  # 64x64 flattened
target = jnp.array([1.0, 0.0])  # Target pattern
model.train(image, target)

# Test on noisy image
noisy_image = image + noise
output = model.inference(noisy_image)

print(f"Confidence: {output[0] / output[1]:.2f}")
```

### Key Files

```
physically_inspired_pattern_matching.py  # Start here
├── PhysicalCrossbar class      # The core model
├── DigitalBaseline class       # For comparison
├── run_benchmark()             # Systematic testing
└── plot_results()              # Visualization

extended_benchmark.py            # Extended analysis
├── Ablation study
├── CNN baseline
└── Theory plots
```

---

## Common Questions

### Q: Do I need a GPU?
A: No. CPU works fine (just slower). GPU speeds things up 5-10x.

### Q: How much disk space needed?
A: About 1.5 GB (mostly for KITTI dataset).

### Q: Can I use my own images?
A: Yes. See `CUSTOM_DATA.md` for guide.

### Q: Why is my digital baseline so bad?
A: That is the point. It shows why physical constraints help.

### Q: What if I get "Out of Memory"?
A: Reduce `N_IMAGES` in the script (default: 100 -> try 20).

---

## Troubleshooting

### Issue: "No module named 'jax'"
```bash
pip install --upgrade jax jaxlib
```

### Issue: "Dataset download fails"
```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### Issue: "CUDA out of memory"
```python
# Reduce batch size or use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
```

---

## Next Steps

After your first run:

1. Check `outputs/` for visualizations
2. Read the full [README.md](README.md)
3. Try the ablation study
4. Modify conductance ranges
5. Share your results

---

## Getting Help

- Bug? Open an issue: https://github.com/or4k2l/truth-seeking-pattern-matching/issues
- Question? Start a discussion: https://github.com/or4k2l/truth-seeking-pattern-matching/discussions
- Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md)

---

Happy experimenting.
