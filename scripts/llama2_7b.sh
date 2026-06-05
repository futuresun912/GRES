#!/bin/bash
# GReS: LLaMA-2-7B experiments
# Reproduces Table III (unstructured + N:M) and Table IV (structured) results

set -e
MODEL="meta-llama/Llama-2-7b-hf"
SAVE="results/llama2_7b"

echo "============================================"
echo "GReS: LLaMA-2-7B"
echo "============================================"

# Unstructured 50%
python main.py --model $MODEL --sparsity_type unstructured --sparsity_ratio 0.5 --save $SAVE/unstructured

# Semi-structured 2:4
python main.py --model $MODEL --sparsity_type 2:4 --save $SAVE/2-4

# Semi-structured 4:8
python main.py --model $MODEL --sparsity_type 4:8 --save $SAVE/4-8

# Structured pruning at 20%
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode head --save $SAVE/struct_head_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode neuron --save $SAVE/struct_neuron_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode gqa_group --save $SAVE/struct_group_20

# Structured pruning at 50%
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.5 --struct_mode head --save $SAVE/struct_head_50
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.5 --struct_mode neuron --save $SAVE/struct_neuron_50
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.5 --struct_mode gqa_group --save $SAVE/struct_group_50

echo "============================================"
echo "LLaMA-2-7B experiments complete"
echo "Results saved to: $SAVE/"
echo "============================================"
