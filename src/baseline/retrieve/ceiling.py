import argparse
import os
import sys

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from conf import (
    CORPUS_DATA_PATH,
    RETRIEVE_RESULT_PATH,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
)
from retrievers import Retriever
from utils import load_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main(opt: argparse.Namespace):
    passage_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "corpus.jsonl")
    if opt.retriever == "e5-base-v2":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "e5-base")
    elif opt.retriever == "contriever":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "contriever")
    else:
        raise NotImplementedError()

    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type=opt.retriever,
    )
    dataset = load_jsonl(
        os.path.join(
            SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH, opt.dataset, "valid.jsonl"
        )
    )

    queries = [
        [chunk["query_info"] for chunk in d["decomposed_questions"].values()]
        # [chunk["sub_question"] for chunk in d["decomposed_questions"].values()]
        for d in dataset
    ]

    chunks = [
        retriever.search(query_chunk, top_k=opt.topk)
        for query_chunk in tqdm_rich(queries, desc="Retrieving")
    ]

    results = []
    for chunk_list, sample in zip(chunks, dataset):
        chunk_list = sum(chunk_list, [])
        chunk_ids = [p["id"] for p in chunk_list]
        sub_ids = sorted(list(sample["decomposed_questions"].keys()))
        oracle_ids = [
            f"{sample['id']}-{sample['decomposed_questions'][sub_id]['positive_paragraph_idx']}"
            for sub_id in sub_ids
        ]

        results.append(
            {
                "question_id": sample["id"],
                "question": sample["question"],
                "oracle_ids": oracle_ids,
                "chunk_ids": chunk_ids,
            }
        )
    output_path = os.path.join(
        RETRIEVE_RESULT_PATH,
        "efficient_rag",
        "filtered_ceiling",
        f"{opt.dataset}-{opt.retriever}-@{opt.topk}.jsonl",
    )
    write_jsonl(results, output_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
