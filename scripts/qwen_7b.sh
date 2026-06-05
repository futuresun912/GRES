#!/bin/bash
# GReS: Qwen2.5-7B experiments (GQA model: 28 heads, 4 KV heads)
set -e
MODEL="Qwen/Qwen2.5-7B"
SAVE="results/qwen_7b"

echo "============================================"
echo "GReS: Qwen2.5-7B (GQA)"
echo "============================================"

python main.py --model $MODEL --sparsity_type unstructured --sparsity_ratio 0.5 --save $SAVE/unstructured
python main.py --model $MODEL --sparsity_type 2:4 --save $SAVE/2-4
python main.py --model $MODEL --sparsity_type 4:8 --save $SAVE/4-8

# Head pruning recommended for Qwen (only 4 KV groups — too coarse for group pruning)
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode head --save $SAVE/struct_head_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode neuron --save $SAVE/struct_neuron_20

echo "Done. Results: $SAVE/"
