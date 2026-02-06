# Contributing to Robust Vision

Thanks for your interest in contributing to Robust Vision!

## Repository

- GitHub: https://github.com/or4k2l/robust-vision
- Issues: https://github.com/or4k2l/robust-vision/issues
- Pull Requests: https://github.com/or4k2l/robust-vision/pulls

## Quick Guidelines

- Keep changes focused and scoped to a single feature or fix.
- Prefer clear, minimal code with reproducible results.
- Add or update documentation when behavior changes.
- Use ASCII text unless there is a clear need for Unicode.

## Development Setup

```bash
git clone https://github.com/or4k2l/robust-vision.git
cd robust-vision
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Suggested Areas

- Additional datasets (Cityscapes, nuScenes)
- Energy or latency analysis
- Hardware experiments on real memristor chips
- Theoretical bounds on robustness gains

## Submitting Changes

1. Create a feature branch.
2. Make your changes with clear commit messages.
3. Open a pull request describing motivation and results.

## Questions

Open an issue with context, logs, and expected behavior.
