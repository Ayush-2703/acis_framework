# Contributing to ACIS Framework

Thank you for your interest in contributing! This framework is the companion
code to a peer-reviewed publication — contributions must maintain academic rigour.

## Ways to Contribute

- **New attack implementations** (e.g. C&W attack, ZOO attack)
- **New defense modules** (e.g. randomised smoothing, TRADES)
- **Construction-specific datasets** (real anonymised data, more scenarios)
- **Empirical validation** (running attacks on real construction AI models)
- **Bug fixes and documentation**

## Setup

```bash
git clone https://github.com/Ayush-2703/acis-framework.git
cd acis-framework
pip install -e ".[dev]"
pre-commit install
```

## Code Standards

- All attacks must subclass `BaseAttack` and return `AttackResult`
- All new threats must reference a published paper in the docstring
- Tests required for all new modules (`pytest tests/ -v`)
- Docstrings follow numpy style

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-attack-name`
3. Add tests for your changes
4. Run `pytest tests/ -v` — all tests must pass
5. Submit a pull request with a clear description

## Academic Attribution

If your contribution introduces a new attack or defense, please cite the
original paper in the docstring and add it to the reference list in `README.md`.
