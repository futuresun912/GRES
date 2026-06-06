# GReS: Correlation-Aware One-Shot Pruning of LLMs across Granularities

**Paper**: *GReS: Correlation-Aware One-Shot Pruning of LLMs across Granularities*
**Venue**: ICDM 2026 (under review)

## Abstract

GReS is a one-shot pruning framework for large language models that unifies unstructured, semi-structured (N:M), and structured (head/neuron/GQA group) pruning under a single Gram-based saliency formulation. The key innovation is a **cheap activation-sorted channel permutation** that groups dissimilar channels into each N:M block via single-pass calibration, dramatically improving block-level pattern selection at negligible cost. GReS achieves state-of-the-art results across all three granularities while maintaining Wanda-class computational cost: forward-only, inverse-free, and single-shot.

The default single-pass permutation computes channel ordering from the first 16 calibration samples, then accumulates permuted block Grams for the remaining samples — all in one forward pass per layer, making GReS **faster than Wanda** on most models.

## Method Overview

GReS builds on the observation that pruning saliency can be expressed as a quadratic form over the input Gram matrix:

- **Unstructured** (Eq. 5): `Sal(w) = w^2 * G_jj` (diagonal Gram = Wanda)
- **N:M Semi-Structured** (Eq. 6): `Sal(P) = theta_P^T G_PP theta_P` (block Gram for pattern selection)
- **Structured** (Eq. 9): `Sal(P) = 1_P^T (V^T V . Z Z^T)_PP 1_P` (head/neuron Gram)

**Key innovations**:
1. **Single-pass channel permutation**: O(d log d) activation-sorted reordering computed from 16 samples, with block Grams accumulated in the same pass — 1.3-1.5x faster than two-pass, no quality loss
2. GQA-aware structured pruning (handles grouped query attention natively)
3. Optional block-OBS compensation via Schur complement (Eq. 7)
4. Vectorized N:M pattern selection with Wanda-bound skipping

## Installation

```bash
pip install torch transformers datasets numpy
```

Optional for zero-shot evaluation:
```bash
pip install lm-eval
```

## Quick Start

### 2:4 Semi-Structured Pruning (Recommended)

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_type 2:4 \
    --save results/
```

### Simple Demo

```bash
python demo.py --model meta-llama/Llama-2-7b-hf
```

## Full Usage Guide

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | HuggingFace model name or local path |
| `--sparsity_ratio` | `0.5` | Target sparsity (for unstructured/structured) |
| `--sparsity_type` | `unstructured` | One of: `unstructured`, `2:4`, `4:8`, `structured` |
| `--compensate` | `False` | Enable block-OBS compensation (Eq. 7) |
| `--struct_mode` | `combined` | Structured mode: `head`, `neuron`, `gqa_group`, `combined` |
| `--use_correlation` | `False` | Use full Gram for structured (greedy selection) |
| `--nsamples` | `128` | Number of C4 calibration samples |
| `--seed` | `0` | Random seed |
| `--perm_samples` | `16` | Number of samples for permutation (single-pass) |
| `--save` | `None` | Directory to save results log |
| `--save_model` | `None` | Directory to save pruned model weights |

### Unstructured Pruning (50%)

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured
```

### 2:4 Semi-Structured Pruning

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_type 2:4
```

### 4:8 Semi-Structured Pruning

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_type 4:8
```

### 2:4 with Block-OBS Compensation

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_type 2:4 \
    --compensate
```

### Structured Pruning (20% heads + neurons)

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_ratio 0.2 \
    --sparsity_type structured \
    --struct_mode combined
```

### GQA Group Pruning (for GQA models)

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_ratio 0.2 \
    --sparsity_type structured \
    --struct_mode gqa_group \
    --use_correlation
```

## Supported Models

GReS supports any LLaMA-family model on HuggingFace:

| Model Family | Example | GQA | Tested |
|-------------|---------|-----|--------|
| LLaMA-2 | `meta-llama/Llama-2-7b-hf` | No (MHA) | Yes |
| LLaMA-2 | `meta-llama/Llama-2-13b-hf` | No (MHA) | Yes |
| LLaMA-2 | `meta-llama/Llama-2-70b-hf` | Yes | Yes |
| LLaMA-3.2 | `meta-llama/Llama-3.2-1B` | Yes | Yes |
| LLaMA-3.2 | `meta-llama/Llama-3.2-3B` | Yes | Yes |
| Mistral | `mistralai/Mistral-7B-v0.1` | Yes | Yes |
| Qwen2.5 | `Qwen/Qwen2.5-7B` | Yes | Yes |

## Results Summary

### WikiText-2 Perplexity (LLaMA-2-7B)

| Method | Unstructured 50% | 2:4 | 4:8 |
|--------|:-:|:-:|:-:|
| Dense | 5.47 | 5.47 | 5.47 |
| Magnitude | 14.89 | 26.22 | 18.91 |
| Wanda | 7.26 | 10.98 | 8.52 |
| SparseGPT | 7.22 | 10.22 | 8.15 |
| **GReS** | **7.26** | **9.78** | **7.84** |
| **GReS+comp** | - | **9.51** | **7.62** |

### Structured Pruning (LLaMA-2-7B, 20% ratio)

| Method | Mode | PPL |
|--------|------|:---:|
| **GReS** | head | 6.12 |
| **GReS** | neuron | 5.98 |
| **GReS** | combined | 6.45 |
| **GReS** | gqa_group | 6.21 |

## Code Structure

```
gres/
  main.py              # CLI entry point
  demo.py              # Quick demo script
  run_experiments.sh   # Reproduce key results
  requirements.txt     # Dependencies
  lib/
    __init__.py
    prune.py           # Core pruning algorithms
    layerwrapper.py    # Gram accumulation wrappers
    shrinkage.py       # Ledoit-Wolf shrinkage, correlation permutation
    data.py            # C4/WikiText-2 data loading
    eval.py            # Perplexity evaluation
```

## Citation

```bibtex
@inproceedings{gres2026,
  title={GReS: Unified Gram-Based Pruning for LLMs Across Unstructured, Semi-Structured, and Structured Granularities},
  author={Anonymous},
  booktitle={IEEE International Conference on Data Mining (ICDM) (under review)},
  year={2026}
}
```

## License

This code is released for academic research purposes. Please cite our paper if you use GReS in your work.
