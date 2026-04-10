#!/bin/bash
# =============================================================================
# AReS: VLM Experiments (Table 2)
# Service Model: CLIP ViT-B/16 | Local Encoder: ViT-B/16
# 16-shot learning setting
# =============================================================================

DATASETS=("flowers102" "dtd" "ucf101" "food101" "gtsrb" "eurosat" "oxfordpets" "stanfordcars" "sun397" "svhn")
SEEDS=(0 1 2)

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "========== ${dataset} | seed=${seed} =========="

        # Stage 1: Prime Once (single-pass API interaction)
        python src/prime_vlm.py \
            --dataset ${dataset} \
            --student vitb16 \
            --mode linear \
            --criterion kl \
            --lr 1e-3 \
            --epochs 100 \
            --num_samples_per_class 16 \
            --seed ${seed}

        # Stage 2: Reprogram Locally (glass-box VR on primed model)
        python src/reprogram.py \
            --dataset ${dataset} \
            --model vitb16 \
            --reprogramming padding \
            --mapping blmp \
            --vlm_distilled \
            --student vitb16 \
            --mode linear \
            --criterion kl \
            --num_samples_per_class 16 \
            --seed ${seed}
    done
done
