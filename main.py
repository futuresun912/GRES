"""
GReS: Gram-based Removal Saliency
One-shot pruning for LLMs across unstructured, semi-structured (N:M), and structured granularities.

Usage examples:
  # Unstructured 50%
  python main.py --model meta-llama/Llama-2-7b-hf --sparsity_ratio 0.5 --sparsity_type unstructured

  # 2:4 semi-structured with channel permutation
  python main.py --model meta-llama/Llama-2-7b-hf --sparsity_type 2:4

  # 2:4 with block-OBS compensation
  python main.py --model meta-llama/Llama-2-7b-hf --sparsity_type 2:4 --compensate

  # 4:8 semi-structured
  python main.py --model meta-llama/Llama-2-7b-hf --sparsity_type 4:8

  # Structured pruning (heads + neurons)
  python main.py --model meta-llama/Llama-2-7b-hf --sparsity_ratio 0.2 --sparsity_type structured

  # GQA group pruning
  python main.py --model meta-llama/Llama-2-7b-hf --sparsity_ratio 0.2 --sparsity_type structured --struct_mode gqa_group
"""

import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.prune import prune_gres, prune_gres_structured, prune_gres_shrink, check_sparsity
from lib.eval import eval_ppl


def get_llm(model_name):
    """
    Load a HuggingFace causal language model with automatic device mapping.

    Args:
        model_name: HuggingFace model name or local path.

    Returns:
        model: Loaded model with model.seqlen set.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="GReS: Gram-based Removal Saliency pruning for LLMs"
    )
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or local path')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for calibration data sampling')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of C4 calibration samples')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5,
                        help='Target sparsity level')
    parser.add_argument('--sparsity_type', type=str, default="unstructured",
                        choices=["unstructured", "4:8", "2:4", "structured"],
                        help='Sparsity granularity')
    parser.add_argument('--compensate', action="store_true",
                        help='Enable block-OBS compensation (Eq. 7)')
    parser.add_argument('--use_correlation', action="store_true",
                        help='Use full Gram for structured pruning (greedy selection)')
    parser.add_argument('--struct_mode', type=str, default="combined",
                        choices=["head", "neuron", "gqa_group", "combined"],
                        help='Structured pruning mode')
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save results log')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Directory to save pruned model weights')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Parse N:M sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type not in ["unstructured", "structured"]:
        assert args.sparsity_ratio == 0.5, \
            "sparsity_ratio must be 0.5 for N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Load model
    model_name = args.model.split("/")[-1]
    print(f"Loading model {args.model}")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if hasattr(model, 'hf_device_map') and "lm_head" in model.hf_device_map:
        device = model.hf_device_map["lm_head"]
    print(f"Using device: {device}")

    # Prune
    if args.sparsity_ratio != 0:
        print("Pruning starts")
        if args.sparsity_type == "structured":
            prune_gres_structured(args, model, tokenizer, device)
        else:
            prune_gres(args, model, tokenizer, device,
                       prune_n=prune_n, prune_m=prune_m)

    # Check sparsity
    print("=" * 40)
    sparsity_ratio = check_sparsity(model)
    print(f"Actual sparsity: {sparsity_ratio:.4f}")
    print("=" * 40)

    # Evaluate
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"WikiText-2 perplexity: {ppl_test:.2f}")

    # Save results
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_filepath = os.path.join(
            args.save, f"log_gres_{args.sparsity_type}.txt"
        )
        with open(save_filepath, "w") as f:
            print("method\tsparsity_type\tactual_sparsity\tppl_test", file=f)
            comp_str = "+comp" if args.compensate else ""
            print(f"gres{comp_str}\t{args.sparsity_type}\t"
                  f"{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f)
        print(f"Results saved to {save_filepath}")

    # Save pruned model
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Pruned model saved to {args.save_model}")


if __name__ == '__main__':
    main()
