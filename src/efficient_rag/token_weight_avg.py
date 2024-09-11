import argparse
import os
import sys
from typing import Literal

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import DebertaV2TokenizerFast as DebertaV2Tokenizer

from conf import (
    EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH,
    EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH,
    TAG_MAPPING_TWO,
)
from efficient_rag.data import FilterDataset, LabelerDataset
from utils import load_jsonl


def build_labeler_dataset(
    dataset: str,
    split: str,
    max_len: int = 384,
    tokenizer=None,
    test_mode: bool = False,
    test_sample_cnt: int = 100,
):
    data_path = os.path.join(
        EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH, dataset, f"{split}.jsonl"
    )
    data = load_jsonl(data_path)
    original_question = [d["question"] for d in data]
    chunk_tokens = [d["chunk_tokens"] for d in data]
    chunk_labels = [d["labels"] for d in data]
    tags = [TAG_MAPPING_TWO[d["tag"]] for d in data]

    if test_mode:
        return LabelerDataset(
            original_question[:test_sample_cnt],
            chunk_tokens[:test_sample_cnt],
            chunk_labels[:test_sample_cnt],
            tags[:test_sample_cnt],
            max_len,
            tokenizer,
        )
    return LabelerDataset(
        original_question, chunk_tokens, chunk_labels, tags, max_len, tokenizer
    )


def build_filter_dataset(
    dataset: str, split: str, max_len: int = 256, tokenizer=None, test_mode=False
):
    data_path = os.path.join(
        EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH, dataset, f"{split}.jsonl"
    )
    data = load_jsonl(data_path)
    texts = [d["query_info_tokens"] for d in data]
    labels = [d["query_info_labels"] for d in data]
    if test_mode:
        return FilterDataset(texts[:100], labels[:100], max_len, tokenizer=tokenizer)
    return FilterDataset(texts, labels, max_len, tokenizer=tokenizer)


def make_dataset(
    type: Literal["filter", "labeler"] = "filter",
    dataset: Literal["musique", "2WikiMQA", "hotpotQA"] = "musique",
    split: str = "train",
    tokenizer: DebertaV2Tokenizer=None,
):
    build_dataset_fn = build_filter_dataset if type == "filter" else build_labeler_dataset
    dataset = build_dataset_fn(dataset, split, tokenizer=tokenizer)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="labeler")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    dataset = make_dataset(args.type, args.dataset, args.split, tokenizer=tokenizer)
    positive_cnt_sum = 0
    negative_cnt_sum = 0
    token_label_key = "token_labels" if args.type == "labeler" else "labels"
    for data in tqdm(dataset):
        attention_mask = data["attention_mask"]
        labels = data[token_label_key]
        positive_cnt = torch.sum(attention_mask & labels).item()
        negative_cnt = torch.sum(attention_mask ^ labels).item()
        positive_cnt_sum += positive_cnt
        negative_cnt_sum += negative_cnt
    print(f"Negative Count for {args.dataset}-{args.split}: {negative_cnt_sum}")
    print(f"Positive Count for {args.dataset}-{args.split}: {positive_cnt_sum}")

    positive_weight = (positive_cnt_sum + negative_cnt_sum) / (2 * positive_cnt_sum)
    negative_weight = (positive_cnt_sum + negative_cnt_sum) / (2 * negative_cnt_sum)
    print(f"Negative Weight for {args.dataset}-{args.split}: {negative_weight}")
    print(f"Positive Weight for {args.dataset}-{args.split}: {positive_weight}")


if __name__ == "__main__":
    main()
