import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tqdm.rich import tqdm_rich

from conf import (
    CONTINUE_TAG,
    EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH,
    EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH,
    FINISH_TAG,
    SYNTHESIZED_NEGATIVE_SAMPLING_EXTRACTED_DATA_PATH,
    TERMINATE_TAG,
)
from utils import load_jsonl, write_jsonl

INFO_TEMPLATE = "Info: {info}"
QUERY_TEMPLATE = "Q: {query}"


def build_labeler_data(samples: list[dict]):
    results = []
    for sample in tqdm_rich(samples, desc="Building labeler data"):
        for subq_id, subq in sample["decomposed_questions"].items():
            try:
                if subq_id == sorted(sample["decomposed_questions"].keys())[-1]:
                    positive_tag = FINISH_TAG
                else:
                    positive_tag = CONTINUE_TAG
                positive_sample = {
                    "question": subq["filtered_query"],
                    "chunk": subq["positive_paragraph"],
                    "matched": subq["matched"],
                    "chunk_tokens": subq["paragraph_tokens"],
                    "labels": subq["labels"],
                    "tag": positive_tag,
                }
                negative_samples = {
                    "question": subq["filtered_query"],
                    "chunk": subq["negative_paragraph"],
                    "matched": subq["negative_matched"],
                    "chunk_tokens": subq["negative_paragraph_tokens"],
                    "labels": subq["negative_labels"],
                    "tag": TERMINATE_TAG,
                }
                results.append(positive_sample)
                results.append(negative_samples)
            except Exception as e:
                continue
    return results


def build_filter_data(samples: list[dict]):
    results = []
    for sample in tqdm_rich(samples, desc="Building filter data"):
        for subq_id, subq in sample["decomposed_questions"].items():
            if "query_info_tokens" not in subq.keys():
                continue
            filter_data = {
                "query_info_tokens": subq["query_info_tokens"],
                "query_info_labels": subq["query_info_labels"],
            }
            results.append(filter_data)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA"],
        required=True,
    )
    parser.add_argument("--split", type=str, default="demo")
    args = parser.parse_args()
    return args


def main(opt: argparse.Namespace):
    data_path = os.path.join(
        SYNTHESIZED_NEGATIVE_SAMPLING_EXTRACTED_DATA_PATH,
        opt.dataset,
        f"{opt.split}.jsonl",
    )
    data = load_jsonl(data_path)

    labeler_data_path = os.path.join(
        EFFICIENT_RAG_LABELER_TRAINING_DATA_PATH, opt.dataset, f"{opt.split}.jsonl"
    )
    labeler_training_data = build_labeler_data(data)
    write_jsonl(labeler_training_data, labeler_data_path)

    filter_data_path = os.path.join(
        EFFICIENT_RAG_FILTER_TRAINING_DATA_PATH, opt.dataset, f"{opt.split}.jsonl"
    )
    filter_training_data = build_filter_data(data)
    write_jsonl(filter_training_data, filter_data_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
