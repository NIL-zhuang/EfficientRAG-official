import argparse
import os
import sys
from typing import Union

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from conf import (
    CORPUS_DATA_PATH,
    EMBEDDING_ALIAS,
    MODEL_DICT,
    SYNTHESIZED_TOKEN_LABELING_DATA_PATH,
)
from language_models import AOAI
from retrievers import Retriever
from retrievers.embeddings import ModelTypes
from utils import load_jsonl


class ChunkSampler:
    def __init__(
        self,
        retriever: Retriever,
    ) -> None:
        self.retriever = retriever

    def sample(self, query: Union[str, list[str]], top_k: int = 10) -> list[list[dict]]:
        def parse_chunk(c):
            return {
                "id": set(
                    [
                        (((pair := cid.split("-"))[0], int(pair[1])) if "-" in cid else (cid, "-1"))
                        for cid in c["id"].split("//")
                    ]
                ),
                "text": c["text"],
            }

        results = self.retriever.search(query, top_k)
        results = [[parse_chunk(c) for c in chunk] for chunk in results]
        return results


def sample_origin_question(sampler: ChunkSampler, dataset: list[dict], top_k: int = 10):
    questions = []
    oracles = []
    for data in dataset:
        questions.append(data["question"])
        oracles.append(
            set([(data["id"], chunk["positive_paragraph_idx"]) for k, chunk in data["decomposed_questions"].items()])
        )
    samples = sampler.sample(questions, top_k)
    coverages = eval(samples, oracles, questions)
    return np.round(np.mean(coverages), 4)


def sample_sub_question(sampler: ChunkSampler, dataset: list[dict], top_k: int = 10):
    questions = []
    oracles = []
    for data in dataset:
        for k, chunk in data["decomposed_questions"].items():
            questions.append(chunk["sub_question"])
            oracles.append(set([(data["id"], chunk["positive_paragraph_idx"])]))
    samples = sampler.sample(questions, top_k)
    coverages = eval(samples, oracles, questions)
    return np.round(np.mean(coverages), 4)


def sample_labeled_words(sampler: ChunkSampler, dataset: list[dict], top_k: int = 10):
    questions = []
    oracles = []
    for data in dataset:
        for k, chunk in data["decomposed_questions"].items():
            labeled_words = " ".join(chunk["labeled_words"])
            questions.append(labeled_words)
            oracles.append(set([(data["id"], chunk["positive_paragraph_idx"])]))
    samples = sampler.sample(questions, top_k)
    coverages = eval(samples, oracles, questions)
    return np.round(np.mean(coverages), 4)


def sample_symmetric_diff(sampler: ChunkSampler, dataset: list[dict], top_k: int = 10):
    def construct_symmetric_diff_query(sample: dict) -> tuple[list[str], list[str]]:
        subq, subo = [], []
        for sub_qid, chunk in data["decomposed_questions"].items():
            # TODO: implement symmetric diff query
            subq.append(chunk["sub_question"])
            subo.append(set([(data["id"], chunk["positive_paragraph_idx"])]))
        return subq, subo

    questions, oracles = [], []
    for data in dataset:
        subq, subo = construct_symmetric_diff_query(data)
        questions.extend(subq)
        oracles.extend(subo)
    samples = sampler.sample(questions, top_k)
    coverages = eval(samples, oracles, questions)
    return np.round(np.mean(coverages), 4)


def eval(samples: list[list[dict]], oracles: list[set], questions: list[str]):
    coverages = []
    for oracle, sample in zip(oracles, samples):
        chunks = set()
        for chunk in sample:
            chunks.update(chunk["id"])
        coverages.append(len(chunks.intersection(oracle)) / len(oracle))
    return coverages


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA"],
        default="musique",
    )
    parser.add_argument("--split", type=str, choices=["train", "valid", "test", "demo"], default="valid")
    parser.add_argument("--embedder", type=str, choices=list(ModelTypes.keys()), help="Embedding Model")
    parser.add_argument("--topk", type=int, default=5, help="Top k retrieval")
    parser.add_argument("--model", choices=["gpt35", "gpt4"], default="gpt4")
    args = parser.parse_args()
    return args


def build_sampler(dataset: str, embedder: str):
    corpus_path = os.path.join(CORPUS_DATA_PATH, dataset)
    passages = os.path.join(corpus_path, "corpus.jsonl")
    embedding_path = os.path.join(corpus_path, EMBEDDING_ALIAS[embedder])
    retriever = Retriever(
        passage_path=passages,
        passage_embedding_path=embedding_path,
        model_type=embedder,
        save_or_load_index=True,
    )
    chunk_sampler = ChunkSampler(retriever)
    return chunk_sampler


def main(opt: argparse.Namespace):
    dataset = load_jsonl(os.path.join(SYNTHESIZED_TOKEN_LABELING_DATA_PATH, opt.dataset, f"{opt.split}.jsonl"))
    chunk_sampler = build_sampler(opt.dataset, opt.embedder)
    for topk in (1, 3, 5, 10, 50, 100, 500, 1000, 2000):
        mean_coverage = sample_origin_question(chunk_sampler, dataset, topk)
        mean_coverage = sample_sub_question(chunk_sampler, dataset, topk)
        mean_coverage = sample_labeled_words(chunk_sampler, dataset, topk)

        print(f"{opt.dataset}-{opt.split}, embedder {opt.embedder}, top-{topk} mean coverage: {mean_coverage}")


if __name__ == "__main__":
    options = parse_args()
    main(options)
