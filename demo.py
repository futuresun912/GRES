"""
GReS Demo: Quick demonstration of 2:4 semi-structured pruning.

Loads a model, runs GReS 2:4 pruning with channel permutation,
evaluates WikiText-2 perplexity, and prints before/after comparison.

Usage:
  python demo.py --model meta-llama/Llama-2-7b-hf
"""

import argparse
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.prune import prune_gres, check_sparsity
from lib.eval import eval_ppl


def main():
    parser = argparse.ArgumentParser(description="GReS 2:4 pruning demo")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='HuggingFace model name or path')
    parser.add_argument('--compensate', action='store_true',
                        help='Enable block-OBS compensation')
    demo_args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    print("=" * 60)
    print("GReS: Gram-based Removal Saliency -- 2:4 Pruning Demo")
    print("=" * 60)

    # Load model
    print(f"\n[1/4] Loading model: {demo_args.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        demo_args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(demo_args.model, use_fast=False)
    device = torch.device("cuda:0")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")
    kv_heads = getattr(model.config, 'num_key_value_heads',
                       model.config.num_attention_heads)
    if kv_heads != model.config.num_attention_heads:
        print(f"  KV heads: {kv_heads} (GQA)")

    # Evaluate dense baseline
    print(f"\n[2/4] Evaluating dense baseline PPL...")
    t0 = time.time()

    class Args:
        pass
    args = Args()
    args.nsamples = 128
    args.seed = 0
    args.sparsity_ratio = 0.5

    ppl_dense = eval_ppl(args, model, tokenizer, device)
    print(f"  Dense PPL: {ppl_dense:.2f} ({time.time() - t0:.1f}s)")

    # Prune with GReS 2:4
    print(f"\n[3/4] Pruning with GReS 2:4 (permuted)...")
    args.compensate = demo_args.compensate
    t0 = time.time()
    prune_gres(args, model, tokenizer, device, prune_n=2, prune_m=4)
    prune_time = time.time() - t0
    print(f"  Pruning took {prune_time:.1f}s")

    sparsity = check_sparsity(model)
    print(f"  Actual sparsity: {sparsity:.4f}")

    # Evaluate pruned model
    print(f"\n[4/4] Evaluating pruned model PPL...")
    t0 = time.time()
    ppl_pruned = eval_ppl(args, model, tokenizer, device)
    print(f"  Pruned PPL: {ppl_pruned:.2f} ({time.time() - t0:.1f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model:          {demo_args.model}")
    print(f"  Sparsity:       2:4 ({sparsity:.1%} actual)")
    print(f"  Compensation:   {'Yes' if demo_args.compensate else 'No'}")
    print(f"  Dense PPL:      {ppl_dense:.2f}")
    print(f"  Pruned PPL:     {ppl_pruned:.2f}")
    print(f"  PPL increase:   {ppl_pruned - ppl_dense:+.2f}")
    print(f"  Pruning time:   {prune_time:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
