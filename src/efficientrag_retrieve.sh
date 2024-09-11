#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
    python src/efficientrag_retrieve.py \
    --dataset hotpotQA \
    --retriever contriever \
    --labels 2 \
    --suffix label2 \
    --labeler_ckpt LABELER_CKPT_PATH \
    --filter_ckpt FILTER_CKPT_PATH
