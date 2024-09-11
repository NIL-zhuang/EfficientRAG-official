import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

from tqdm.rich import tqdm_rich

from conf import (
    CORPUS_DATA_PATH,
    SYNTHESIZED_NEGATIVE_SAMPLING_DATA_PATH,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
)
from retrievers import Retriever
from retrievers.embeddings import ModelCheckpointMapping, ModelTypes
from utils import load_jsonl


def negative_sampling(retriever: Retriever, samples: list[dict]) -> Iterator[dict]:
    for sample in tqdm_rich(samples, total=len(samples), desc="Negative Sampling..."):
        if not all(
            ["filtered_query" in sample["decomposed_questions"][sub_id] for sub_id in sample["decomposed_questions"]]
        ):
            print(f"Invalid sample {sample['id']}")
            continue
        sub_ids = sorted(list(sample["decomposed_questions"].keys()))
        filtered_queries = [
            sample["decomposed_questions"][sub_id]["filtered_query"]
            for sub_id in sub_ids
            if sample["decomposed_questions"][sub_id]
        ]
        oracle_chunk_ids = set(
            [
                f"{sample['id']}-{'{:02d}'.format(sample['decomposed_questions'][sub_id]['positive_paragraph_idx'])}"
                for sub_id in sub_ids
            ]
        )
        result = sample.copy()
        candidate_chunks = retriever.search(filtered_queries, top_k=10)
        for subq_id, candidate_chunk_list in zip(sub_ids, candidate_chunks):
            for candidate_chunk in candidate_chunk_list:
                chunk_idx = set(candidate_chunk["id"].split("//"))
                if not chunk_idx.intersection(oracle_chunk_ids):
                    result["decomposed_questions"][subq_id]["negative_paragraph"] = candidate_chunk["text"]
                    result["decomposed_questions"][subq_id]["negative_paragraph_idx"] = candidate_chunk["id"]
                    break
        yield result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["musique", "2WikiMQA", "hotpotQA"],
        required=True,
    )
    parser.add_argument("--split", type=str, default="demo")
    parser.add_argument("--retriever", type=str, choices=ModelTypes.keys(), default="contriever")
    args = parser.parse_args()
    return args


def main(opts: argparse.Namespace):
    passage_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, "corpus.jsonl")
    if opts.retriever == "e5-base-v2":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, "e5-base")
    elif opts.retriever == "contriever":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, "contriever")
    else:
        raise NotImplementedError(f"Retriever {opts.retriever} not implemented")

    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type=opts.retriever,
        model_path=ModelCheckpointMapping[opts.retriever],
    )
    subq_data_path = os.path.join(SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH, opts.dataset, f"{opts.split}.jsonl")
    samples = load_jsonl(subq_data_path)
    output_data_path = os.path.join(SYNTHESIZED_NEGATIVE_SAMPLING_DATA_PATH, opts.dataset, f"{opts.split}.jsonl")
    with open(output_data_path, "w+", encoding="utf-8") as f:
        for sample in negative_sampling(retriever, samples):
            d = json.dumps(sample, ensure_ascii=False)
            f.write(d + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
