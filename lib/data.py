"""
Data loading utilities for GReS calibration and evaluation.

Supports C4 (calibration) and WikiText-2 (evaluation) datasets.
"""

import random
import numpy as np
import torch
from datasets import load_dataset


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


class TokenizerWrapper:
    """Minimal wrapper to hold tokenized input IDs for evaluation."""
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load WikiText-2 dataset for calibration and evaluation.

    Args:
        nsamples: Number of calibration samples to draw.
        seed: Random seed for reproducible sample selection.
        seqlen: Sequence length for each sample.
        tokenizer: HuggingFace tokenizer.

    Returns:
        trainloader: List of (input_ids, targets) tuples for calibration.
        testloader: TokenizerWrapper containing the full test set.
    """
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load C4 dataset for calibration and evaluation.

    Uses streaming to avoid downloading the full dataset.

    Args:
        nsamples: Number of calibration samples to draw.
        seed: Random seed for reproducible sample selection.
        seqlen: Sequence length for each sample.
        tokenizer: HuggingFace tokenizer.

    Returns:
        trainloader: List of (input_ids, targets) tuples for calibration.
        valloader: TokenizerWrapper containing validation data.
    """
    traindata = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    valdata = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

    random.seed(seed)
    trainloader = []
    train_iter = iter(traindata.shuffle(seed=seed, buffer_size=10000))
    for _ in range(nsamples):
        while True:
            sample = next(train_iter)
            trainenc = tokenizer(sample['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    val_texts = []
    val_iter = iter(valdata)
    for _ in range(1100):
        try:
            val_texts.append(next(val_iter)['text'])
        except StopIteration:
            break
    valenc = tokenizer(' '.join(val_texts), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Get data loaders by dataset name.

    Args:
        name: Dataset name ('c4' or 'wikitext2').
        nsamples: Number of calibration samples.
        seed: Random seed.
        seqlen: Sequence length.
        tokenizer: HuggingFace tokenizer.

    Returns:
        trainloader: Calibration data.
        testloader: Evaluation data.
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset: {name}. Supported: 'c4', 'wikitext2'.")
