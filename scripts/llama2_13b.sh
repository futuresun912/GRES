#!/bin/bash
# GReS: LLaMA-2-13B experiments
set -e
MODEL="meta-llama/Llama-2-13b-hf"
SAVE="results/llama2_13b"

echo "============================================"
echo "GReS: LLaMA-2-13B"
echo "============================================"

python main.py --model $MODEL --sparsity_type unstructured --sparsity_ratio 0.5 --save $SAVE/unstructured
python main.py --model $MODEL --sparsity_type 2:4 --save $SAVE/2-4
python main.py --model $MODEL --sparsity_type 4:8 --save $SAVE/4-8

python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode head --save $SAVE/struct_head_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.2 --struct_mode neuron --save $SAVE/struct_neuron_20
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.5 --struct_mode head --save $SAVE/struct_head_50
python main.py --model $MODEL --sparsity_type structured --sparsity_ratio 0.5 --struct_mode neuron --save $SAVE/struct_neuron_50

echo "Done. Results: $SAVE/"
