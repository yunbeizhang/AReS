#!/bin/bash
# =============================================================================
# AReS: Quick Example on Flowers102
# This script runs the full AReS pipeline (VLM setting) on Flowers102
# Service Model: CLIP ViT-B/16 | Local Encoder: ViT-B/16 | 16-shot
# Expected accuracy: ~86.6% (Table 2)
# =============================================================================

DATASET="flowers102"
SEED=0

echo "===== Step 1: Prime Once (Single-pass API interaction) ====="
echo "Training a linear layer on ViT-B/16 using CLIP ViT-B/16 predictions..."
python src/prime_vlm.py \
    --dataset ${DATASET} \
    --student vitb16 \
    --mode linear \
    --criterion kl \
    --lr 1e-3 \
    --epochs 100 \
    --num_samples_per_class 16 \
    --seed ${SEED}

echo ""
echo "===== Step 2: Reprogram Locally (Glass-box VR on primed model) ====="
echo "Training visual prompt on primed local model..."
python src/reprogram.py \
    --dataset ${DATASET} \
    --model vitb16 \
    --reprogramming padding \
    --mapping blmp \
    --vlm_distilled \
    --student vitb16 \
    --mode linear \
    --criterion kl \
    --num_samples_per_class 16 \
    --seed ${SEED}

echo ""
echo "===== Done! ====="
echo "Results saved to ./results_vlm/"
