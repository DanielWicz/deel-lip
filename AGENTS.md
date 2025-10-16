# AGENTS Guide

This document gives a quick tour of the repository so you can orient yourself quickly. Paths are relative to the project root.

## Root Files

- `README.md` — Intro to deel-lip, installation notes, badges, and link to documentation.
- `LICENSE` — MIT license for the project.
- `CONTRIBUTING.md` — Guidelines for people who want to contribute.
- `Makefile` — Convenience commands (docs build, lint, etc.).
- `setup.py` / `setup.cfg` — Packaging metadata; `setup.py` now depends on PyTorch.
- `MANIFEST.in` — Which extra files to include in the built package.
- `mkdocs.yml` — Configuration for the documentation site.
- `.gitignore` — Files that shouldn’t be committed.

## Python Package (`deel/lip`)

Core PyTorch implementation of Lipschitz-aware tools.

- `__init__.py` — Exposes public API, loads version.
- `VERSION` — Single-source package version string.
- `activations.py` — Torch activations that control Lipschitz constants (GroupSort, Householder, etc.).
- `callbacks.py` — Lightweight callback classes (condensation, monitoring, loss-parameter scheduling).
- `compute_layer_sv.py` — Utilities to estimate singular values of modules.
- `constraints.py` — Weight constraints and projections (spectral, Frobenius, clipping).
- `initializers.py` — Spectral initializer implemented with torch.
- `losses.py` — Suite of KR/HKR/hinge/tau losses backed by `torch.nn.Module`.
- `metrics.py` — Provable robustness metrics implemented in PyTorch.
- `model.py` — Lipschitz-aware `Sequential` and `Model` abstractions.
- `normalizers.py` — Spectral/Björck normalization routines on tensors.
- `regularizers.py` — Orthogonality regularizers for dense and convolutional weights.
- `utils.py` — Helper functions (Jacobian Lipschitz estimate, padding, label processing).
- `layers/` — Subpackage containing
  - `base_layer.py` — Mixins for Lipschitz-aware modules.
  - `dense.py`, `convolutional.py`, `pooling.py`, `unconstrained.py` — Torch layers (spectral/frobenius dense & conv, pooling, padded conv helper).
  - `activations.py` — Activation layers used internally.
  - `__init__.py` — Exports layer classes.

## Tests (`tests/`)

PyTorch-based unit tests covering the main components.

- `test_activations.py` — Sanity checks on custom activations.
- `test_compute_layer_sv.py` — Confirms spectral dense layers yield unit singular values.
- `test_condense.py` — Verifies condensation callback writes constrained weights.
- `test_initializers.py` — Ensures spectral initializer creates orthogonal matrices.
- `test_layers.py` — Basic forward passes for spectral dense/conv layers.
- `test_losses.py` — Smoke tests for KR/HKR and tau losses.
- `test_metrics.py` — Validates robustness metrics output scalars.
- `test_models.py` — Checks sequential Lipschitz propagation and vanilla export.
- `test_normalization.py` / `test_normalizers.py` — Routines for Björck and spectral normalization.
- `test_pooling.py` — Tests scaled pooling layers and invertible up/down sampling.
- `test_regularizers.py` — Covers Lorth and dense orthogonal regularizers.
- `test_residual.py` — Basic gradient stability check through sequential model.
- `test_unconstrained_layers.py` — Exercises `PadConv2D` padding modes.
- `test_updownsampling.py` — Round-trip test for invertible sampling.
- `utils_framework.py` — Minimal helpers for tests (seed control, tensor conversions).

## Documentation (`docs/`)

- `index.md` — Landing page for mkdocs site.
- `api/` — Auto-generated API reference stubs for modules.
- `assets/` — Images used by README/docs.
- `css/`, `js/` — Custom styling/scripts.
- `notebooks/` — Jupyter notebooks for tutorials/demos.

