# Physically-Inspired Robust Pattern Matching

**High-Confidence Pattern Recognition via Memristive Crossbar Arrays**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX 0.4+](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/or4k2l/physical-pattern-matching/blob/main/notebook.ipynb)

**Full paper:** [paper.md](paper.md) or [paper.tex](paper.tex)

---

## Overview

This repository presents a physically-inspired approach to robust pattern recognition using memristive crossbar arrays. The key finding is that **physical constraints provide implicit regularization**, yielding much higher confidence margins than standard digital approaches under noise.

## ⚠️ Important Note

**This is exploratory research. The central claim that physical
constraints improve robustness is NOT conclusively proven.**

Our ablation study shows that unconstrained Hebbian learning
achieves HIGHER SNR than constrained versions. The advantage
over CNNs may stem from architectural differences rather than
physical constraints.

We're sharing this work for transparency and to invite community
feedback. **Treat conclusions with caution.**

**Key result:** Physical crossbars achieve **158x higher confidence margins** than CNNs at equal accuracy on KITTI.

| Metric | Physical Crossbar | Digital Baseline | CNN |
|--------|------------------|------------------|-----|
| **Accuracy** | 100.0% | 35.0% | 100.0% |
| **Mean SNR** | 171.18 | 1.93 | 1.08 |

---

## Quick Start

Install dependencies and run the benchmark:

```bash
git clone https://github.com/or4k2l/physical-pattern-matching.git
cd physical-pattern-matching
pip install -r requirements.txt
python physically_inspired_pattern_matching.py
```

Extended benchmark (ablation + CNN baseline):

```bash
python extended_benchmark.py
```

More details are in [quick_start_guide.md](quick_start_guide.md).

---

## Results
![Comprehensive comparison](assets/comprehensive_comparison.png)
Comprehensive comparison across noise levels (physical vs digital vs CNN).

![Ablation analysis](assets/ablation_analysis.png)
Conductance range ablation: robustness, accuracy, saturation, and weight variability.

![Theoretical analysis](assets/theoretical_analysis.png)
Theory visuals for saturation, noise sensitivity, and implicit regularization.

---

## Code Structure

```
.
├── physically_inspired_pattern_matching.py  # Main benchmark (10 images)
├── extended_benchmark.py                    # Extended version (100 images)
├── requirements.txt                         # Python dependencies
├── paper.md                                 # Full paper (Markdown)
├── paper.tex                                # Full paper (IEEE LaTeX)
└── assets/                                  # Result visualizations
```

---

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.

---

## Contributing

Ideas that would help most:
- Additional datasets (Cityscapes, nuScenes)
- Energy or latency analysis
- Hardware experiments on real memristor chips
- Theoretical bounds on robustness gains

---

## Links

- Quick start: [quick_start_guide.md](quick_start_guide.md)
- Colab notebook: [notebook.ipynb](notebook.ipynb)
- Custom data: [CUSTOM_DATA.md](CUSTOM_DATA.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Submission guide: [submission_guide.md](submission_guide.md)
- GitHub setup: [github_setup_checklist.md](github_setup_checklist.md)
- Issues: https://github.com/or4k2l/physical-pattern-matching/issues
- Discussions: https://github.com/or4k2l/physical-pattern-matching/discussions





