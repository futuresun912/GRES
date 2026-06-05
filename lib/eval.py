"""
Perplexity evaluation for pruned models.

Evaluates on WikiText-2 test set using sliding window with model.seqlen.
"""

import torch
import torch.nn as nn
from .data import get_loaders


@torch.no_grad()
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Evaluate WikiText-2 perplexity.

    Args:
        args: Arguments object (unused, kept for API compatibility).
        model: HuggingFace causal LM with model.seqlen attribute.
        tokenizer: HuggingFace tokenizer.
        device: Torch device for evaluation.

    Returns:
        ppl: Float perplexity value.
    """
    dataset = "wikitext2"
    print(f"Evaluating perplexity on {dataset}")
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    ppl_test = eval_ppl_wikitext(model, testloader, bs=1, device=device)
    return ppl_test


@torch.no_grad()
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    """
    Compute perplexity on a tokenized test set.

    Processes the test set in non-overlapping windows of model.seqlen tokens,
    accumulating cross-entropy loss to compute perplexity.

    Args:
        model: HuggingFace causal LM with model.seqlen attribute.
        testenc: TokenizerWrapper with .input_ids attribute.
        bs: Batch size for evaluation.
        device: Torch device.

    Returns:
        ppl: Float perplexity value.
    """
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    nlls = []
    print(f"Evaluating {nsamples} samples")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"  sample {i}/{nsamples}")
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()
    return ppl.item()
