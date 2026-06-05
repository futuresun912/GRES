#!/bin/bash
# GReS: LLaMA-3.2-3B experiments (GQA model: 24 heads, 8 KV heads)
set -e
MODEL="meta-llama/Llama-3.2-3B"
SAVE="results/llama3_3b"

echo "============================================"
echo "GReS: LLaMA-3.2-3B (GQA)"
echo "============================================"

python main.py --model $MODEL --sparsity_type unstructured --sparsity_ratio 0.5 --save $SAVE/unstructured
python main.py --model $MODEL --sparsity_type 2:4 --save $SAVE/2-4
python main.py --model $MODEL --sparsity_type 4:8 --save $SAVE/4-8

# GQA group pruning is the recommended variant for GQA models
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode head --save $SAVE/struct_head_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode neuron --save $SAVE/struct_neuron_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode gqa_group --save $SAVE/struct_group_20

echo "Done. Results: $SAVE/"
