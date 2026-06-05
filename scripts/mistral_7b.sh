#!/bin/bash
# GReS: Mistral-7B experiments (GQA model: 32 heads, 8 KV heads)
set -e
MODEL="mistralai/Mistral-7B-v0.1"
SAVE="results/mistral_7b"

echo "============================================"
echo "GReS: Mistral-7B (GQA)"
echo "============================================"

python main.py --model $MODEL --sparsity_type unstructured --sparsity_ratio 0.5 --save $SAVE/unstructured
python main.py --model $MODEL --sparsity_type 2:4 --save $SAVE/2-4
python main.py --model $MODEL --sparsity_type 4:8 --save $SAVE/4-8

python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode head --save $SAVE/struct_head_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode neuron --save $SAVE/struct_neuron_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode gqa_group --save $SAVE/struct_group_20

echo "Done. Results: $SAVE/"
