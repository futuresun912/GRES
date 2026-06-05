#!/bin/bash
# GReS: Reproduce key results from the paper
# Default model: LLaMA-2-7B (change MODEL below for other models)
#
# Usage:
#   bash run_experiments.sh           # Run all experiments
#   bash run_experiments.sh 2:4       # Run only 2:4 experiments
#   bash run_experiments.sh structured # Run only structured experiments

set -e

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
SAVE_DIR="${SAVE_DIR:-results}"

echo "============================================"
echo "GReS Experiments"
echo "Model: ${MODEL}"
echo "Results: ${SAVE_DIR}"
echo "============================================"

run_section() {
    local section=$1
    [[ -z "$TARGET" || "$TARGET" == "$section" ]]
}

TARGET="${1:-}"

# --- Unstructured 50% ---
if run_section "unstructured"; then
    echo ""
    echo ">>> Unstructured 50% pruning"
    python main.py \
        --model ${MODEL} \
        --sparsity_ratio 0.5 \
        --sparsity_type unstructured \
        --save ${SAVE_DIR}
fi

# --- 2:4 Semi-Structured ---
if run_section "2:4"; then
    echo ""
    echo ">>> GReS 2:4 pruning"
    python main.py \
        --model ${MODEL} \
        --sparsity_type 2:4 \
        --save ${SAVE_DIR}

    echo ""
    echo ">>> GReS 2:4 + compensation"
    python main.py \
        --model ${MODEL} \
        --sparsity_type 2:4 \
        --compensate \
        --save ${SAVE_DIR}
fi

# --- 4:8 Semi-Structured ---
if run_section "4:8"; then
    echo ""
    echo ">>> GReS 4:8 pruning"
    python main.py \
        --model ${MODEL} \
        --sparsity_type 4:8 \
        --save ${SAVE_DIR}

    echo ""
    echo ">>> GReS 4:8 + compensation"
    python main.py \
        --model ${MODEL} \
        --sparsity_type 4:8 \
        --compensate \
        --save ${SAVE_DIR}
fi

# --- Structured Pruning ---
if run_section "structured"; then
    echo ""
    echo ">>> Structured: head pruning (20%)"
    python main.py \
        --model ${MODEL} \
        --sparsity_ratio 0.2 \
        --sparsity_type structured \
        --struct_mode head \
        --use_correlation \
        --save ${SAVE_DIR}

    echo ""
    echo ">>> Structured: neuron pruning (20%)"
    python main.py \
        --model ${MODEL} \
        --sparsity_ratio 0.2 \
        --sparsity_type structured \
        --struct_mode neuron \
        --save ${SAVE_DIR}

    echo ""
    echo ">>> Structured: GQA group pruning (20%)"
    python main.py \
        --model ${MODEL} \
        --sparsity_ratio 0.2 \
        --sparsity_type structured \
        --struct_mode gqa_group \
        --use_correlation \
        --save ${SAVE_DIR}

    echo ""
    echo ">>> Structured: combined head+neuron (20%)"
    python main.py \
        --model ${MODEL} \
        --sparsity_ratio 0.2 \
        --sparsity_type structured \
        --struct_mode combined \
        --use_correlation \
        --save ${SAVE_DIR}

    echo ""
    echo ">>> Structured: combined + compensation (20%)"
    python main.py \
        --model ${MODEL} \
        --sparsity_ratio 0.2 \
        --sparsity_type structured \
        --struct_mode combined \
        --use_correlation \
        --compensate \
        --save ${SAVE_DIR}
fi

echo ""
echo "============================================"
echo "All experiments complete. Results in ${SAVE_DIR}/"
echo "============================================"
