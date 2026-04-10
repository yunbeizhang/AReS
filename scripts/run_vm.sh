#!/bin/bash
# =============================================================================
# AReS: VM Experiments (Table 3)
# Service Model: ViT-B/16 | Local Encoder: ViT-B/32 or RN50
# Full-shot learning setting
# =============================================================================

DATASETS=("flowers102" "dtd" "ucf101" "food101" "gtsrb" "eurosat" "oxfordpets" "stanfordcars" "sun397" "svhn")
SEEDS=(0 1 2)
LOCAL_ENCODER="vitb32"  # or "resnet50"
SERVICE_MODEL="vitb16"

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "========== ${dataset} | seed=${seed} =========="

        # Stage 1: Prime Once (knowledge transfer from service to local)
        python src/prime_vm.py \
            --dataset ${dataset} \
            --teacher ${SERVICE_MODEL} \
            --student ${LOCAL_ENCODER} \
            --mode linear \
            --criterion kl \
            --lr 1e-3 \
            --epochs 100 \
            --seed ${seed}

        # Stage 2: Reprogram Locally (glass-box VR on primed model)
        python src/reprogram.py \
            --dataset ${dataset} \
            --model ${LOCAL_ENCODER} \
            --reprogramming padding \
            --mapping blmp \
            --distilled \
            --teacher ${SERVICE_MODEL} \
            --student ${LOCAL_ENCODER} \
            --mode linear \
            --criterion kl \
            --seed ${seed}
    done
done
