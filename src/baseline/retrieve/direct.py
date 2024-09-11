import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import time

from tqdm.rich import tqdm_rich

from conf import (
    CORPUS_DATA_PATH,
    RETRIEVE_RESULT_PATH,
    SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
)
from language_models import LanguageModel, get_model
from retrievers import Retriever
from utils import ask_model, load_jsonl, write_jsonl

DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA = """
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
<Answer>:
```json
{{"answer": "The Mask Of Fu Manchu"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
<Answer>:
```json
{{"answer": "12 June 1516"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
<Answer>:
```json
{{"answer": "El Extrano Viaje"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()

DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE = """
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: In which year did the publisher of In Cold Blood form?
<Answer>:
```json
{{"answer": "2001"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
<Answer>:
```json
{{"answer": "John Cranley"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
<Answer>:
```json
{{"answer": "eastern tip"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()

DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA = """
Answer the given question in JSON format, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
<Answer>:
```json
{{"answer": "Little Richard"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{"answer": "Chinua Achebe"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{"answer": "1979"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model", type=str, default="llama-8B")
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def process_sample(
    model: LanguageModel,
    retriever: Retriever,
    prompt_template: str,
    sample: dict,
    topk: int = 10,
) -> dict:
    question = sample["question"]
    knowledge = retriever.search(question, top_k=topk)[0]
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
    output_path = os.path.join(
        RETRIEVE_RESULT_PATH, "direct", f"{opt.dataset}-@{opt.topk}.jsonl"
    )
    write_jsonl(results, output_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
