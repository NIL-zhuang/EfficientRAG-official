import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import time

from direct import (
    DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
    DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
    DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
)
from tqdm.rich import tqdm_rich

from conf import (
    CORPUS_DATA_PATH,
    RETRIEVE_RESULT_PATH,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
)
from empirical_retrieve import QUERY_DECOMPOSE_PROMPT
from language_models import LanguageModel, get_model
from retrievers import Retriever
from utils import ask_model, load_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model", type=str, default="llama-8B")
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def llm_decompose(model, question):
    prompt = QUERY_DECOMPOSE_PROMPT.format(question=question)
    decomposition = ask_model(
        model,
        prompt,
        type="json",
        check_if_valid=lambda x: type(x) is dict and "decomposed_questions" in x,
    )
    queries = decomposition["decomposed_questions"]
    return queries


def process_sample(
    model: LanguageModel,
    retriever: Retriever,
    prompt_template: str,
    sample: dict,
    topk: int = 10,
) -> dict:
    question = sample["question"]
    decomposed_questions = llm_decompose(model, question)
    # TODO: noqa, check if it works!!
    knowledges = sum(retriever.search(decomposed_questions, top_k=topk), [])

    # deduplicate knowledge
    seen_ids = set()
    knowledge = []
    for k in knowledges:
        if k["id"] not in seen_ids:
            knowledge.append(k)
            seen_ids.add(k["id"])

    # knowledge = retriever.search(question, top_k=topk)[0]
    knowledge_chunk = "\n".join([p["text"] for p in knowledge])
    chunk_ids = [p["id"] for p in knowledge]
    prompt = prompt_template.format(knowledge=knowledge_chunk, question=question)
    result = ask_model(
        model,
        prompt,
        type="json",
        check_if_valid=lambda x: type(x) is dict and "answer" in x,
        mode="chat",
    )
    sub_ids = sorted(list(sample["decomposed_questions"].keys()))
    oracle_ids = [
        f"{sample['id']}-{'{:02d}'.format(sample['decomposed_questions'][sub_id]['positive_paragraph_idx'])}"
        for sub_id in sub_ids
    ]
    result = {
        "question_id": sample["id"],
        "question": question,
        "answer": sample["answer"],
        "oracle_ids": oracle_ids,
        "chunk_ids": chunk_ids,
        "model_output": result["answer"],
    }
    return result


def process_dataset(
    model: LanguageModel,
    retriever: Retriever,
    prompt_template: str,
    topk: int,
    dataset: List[dict],
    max_workers: int = 10,
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {
            executor.submit(
                process_sample, model, retriever, prompt_template, sample, topk
            ): idx
            for idx, sample in enumerate(dataset)
        }
        results = []
        for future in tqdm_rich(as_completed(tasks), total=len(dataset)):
            idx = tasks[future]
            try:
                res = future.result()
                results.append((idx, res))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
    results = [r[1] for r in sorted(results, key=lambda x: x[0])]
    return results


def main(opt: argparse.Namespace):
    start = time.time()
    passage_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "corpus.jsonl")
    if opt.retriever == "e5-base-v2":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "e5-base")
    elif opt.retriever == "contriever":
        embedding_path = os.path.join(CORPUS_DATA_PATH, opt.dataset, "contriever")
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

    prompt_template_mapping = {
        "musique": DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
        "2WikiMQA": DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
        "hotpotQA": DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
    }
    prompt_template = prompt_template_mapping[opt.dataset]
    model: LanguageModel
    model = get_model(opt.model)
    results = process_dataset(
        model, retriever, prompt_template, opt.topk, dataset, max_workers=opt.workers
    )

    end = time.time()
    print(f"Retrieval time: {end - start:.2f}s")
    print(f"Average retrieval time: {(end - start) / len(dataset):.2f}s")
    output_dir = os.path.join(RETRIEVE_RESULT_PATH, "decomposed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{opt.dataset}-@{opt.topk}.jsonl")
    write_jsonl(results, output_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
