import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from prompts.span_labeling import SPAN_LABELING_PROMPT, SPAN_LABELING_SYSTEM_PROMPT

from conf import (
    MODEL_DICT,
    SYNTHESIZED_DECOMPOSED_DATA_PATH,
    SYNTHESIZED_SPAN_LABELING_DATA_PATH,
)
from language_models import AOAI, LlamaServer
from utils import ask_model, load_jsonl

BEGIN_OF_QUESTION_SPAN_TOKEN = "<q-span>"
END_OF_QUESTION_SPAN_TOKEN = "</q-span>"
BEGIN_OF_ANSWER_SPAN_TOKEN = "<a-span>"
END_OF_ANSWER_SPAN_TOKEN = "</a-span>"

QUESTION_SPAN_TOKEN_PATTERN = (
    rf"{BEGIN_OF_QUESTION_SPAN_TOKEN}(.+?){END_OF_QUESTION_SPAN_TOKEN}"
)
ANSWER_SPAN_TOKEN_PATTERN = (
    rf"{BEGIN_OF_ANSWER_SPAN_TOKEN}(.+?){END_OF_ANSWER_SPAN_TOKEN}"
)


class SpanLabeler:
    def __init__(self, model: str, dataset: str, split: str) -> None:
        if "gpt" in model:
            self.model = AOAI(model)
        elif "Llama" in model:
            self.model = LlamaServer(model)

        decomposed_data_path = os.path.join(
            SYNTHESIZED_DECOMPOSED_DATA_PATH, dataset, f"{split}.jsonl"
        )
        self.data = load_jsonl(decomposed_data_path)
        self.data_mapping = {d["id"]: d for d in self.data}
        # self.check_if_valid = lambda x: True
        self.check_if_valid = (
            lambda x: all(
                [
                    k in x.keys()
                    for k in [
                        "labeled_question",
                        "labeled_document",
                    ]
                ]
            )
            and x["labeled_question"].find(BEGIN_OF_QUESTION_SPAN_TOKEN) >= 0
            and x["labeled_question"].find(END_OF_QUESTION_SPAN_TOKEN) >= 0
            and x["labeled_document"].find(BEGIN_OF_ANSWER_SPAN_TOKEN) >= 0
            and x["labeled_document"].find(END_OF_ANSWER_SPAN_TOKEN) >= 0
        )

    def parse(self, starting: int = 0, ending: int = None, workers=10) -> list[dict]:
        if ending is None:
            ending = len(self.data)
        data = self.data[starting:ending]
        data = [d for d in data if d.get("state", None) is None]

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tasks = {
                    executor.submit(self.parse_sample, sample): idx
                    for idx, sample in enumerate(data)
                }
                results = []
                for future in tqdm_rich(
                    as_completed(tasks), total=len(tasks), desc="Processing..."
                ):
                    task_id = tasks[future]
                    try:
                        result = future.result()
                        results.append((task_id, result))
                    finally:
                        ...
            results = [result[1] for result in sorted(results, key=lambda x: x[0])]
        else:
            results = [
                self.parse_sample(sample)
                for sample in tqdm_rich(data, desc="Processing...")
            ]
        return results

    def parse_sample(
        self,
        sample: dict,
    ) -> dict:
        # First hop
        for subq_id, chunk in sample["decomposed_questions"].items():
            if len(chunk["dependency"]) == 0:
                chunk["current_question"] = sample["question"]

        max_iter = 5
        cur_iter = 0
        while cur_iter < max_iter and not all(
            [
                "current_question" in subq.keys()
                for subq in sample["decomposed_questions"].values()
            ]
        ):
            cur_iter += 1
            # construct filtered_query for each sub-question
            prompt_list, subq_id_list, current_question_list = self.parse_prompt(sample)
            if len(prompt_list) == 0:
                break

            for prompt, subq_id, current_question in zip(
                prompt_list, subq_id_list, current_question_list
            ):
                result = ask_model(
                    self.model,
                    prompt,
                    SPAN_LABELING_SYSTEM_PROMPT,
                    type="json",
                    check_if_valid=self.check_if_valid,
                    sleep=False,
                )
                chunk = sample["decomposed_questions"][subq_id]
                chunk["current_question"] = current_question
                try:
                    assert "span_status" not in chunk.keys(), "The chunk parse failed"
                    assert result is not None, "result is None"
                    result = self.parse_result(result)
                except Exception as e:
                    print(e)
                    chunk["span_status"] = "error"
                    continue

                for k, v in result.items():
                    chunk[k] = v

        return sample

    def parse_result(self, result: dict) -> dict:
        labeled_question = result["labeled_question"]
        labeled_document = result["labeled_document"]

        part_of_question = re.search(
            QUESTION_SPAN_TOKEN_PATTERN, labeled_question
        ).group(1)
        part_of_document = re.search(ANSWER_SPAN_TOKEN_PATTERN, labeled_document).group(
            1
        )
        next_question = (
            labeled_question.replace(part_of_question, part_of_document)
            .replace(BEGIN_OF_QUESTION_SPAN_TOKEN, "")
            .replace(END_OF_QUESTION_SPAN_TOKEN, "")
        )
        return {
            "labeled_question": labeled_question,
            "labeled_document": labeled_document,
            "part_of_question": part_of_question,
            "part_of_document": part_of_document,
            "next_question": next_question,
        }

    def parse_prompt(self, sample: dict) -> list[dict]:
        def construct_subq(subq_id, dependencies):
            if len(dependencies) == 0:
                return sample["question"]

            cur_question = None
            for dep in dependencies:
                if dep in sample["decomposed_questions"].keys():
                    subq = sample["decomposed_questions"][dep]
                    part_of_question = subq.get("part_of_question", None)
                    part_of_document = subq.get("part_of_document", None)
                    if part_of_question is None or part_of_document is None:
                        return None

                    if cur_question is None:
                        cur_question = subq["current_question"]
                    cur_question = cur_question.replace(
                        part_of_question, part_of_document
                    )
            return cur_question

        prompt_list = []
        subq_id_list = []
        current_question_list = []

        for subq_id in sorted(sample["decomposed_questions"].keys()):
            subq = sample["decomposed_questions"][subq_id]
            deps = subq["dependency"]
            if "next_question" in subq.keys():
                # already labeled
                continue

            question = construct_subq(subq_id, deps)
            if question is None:
                # not ready to be labeled
                continue

            sub_question = subq["sub_question"]
            paragraph = subq["positive_paragraph"]
            sub_answer = subq["answer"]

            prompt = SPAN_LABELING_PROMPT.format(
                multi_hop_question=question,
                single_hop_question=sub_question,
                document=paragraph,
                answer=sub_answer,
            )
            prompt_list.append(prompt)
            subq_id_list.append(subq_id)
            current_question_list.append(question)
        return prompt_list, subq_id_list, current_question_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA"],
        default="musique",
    )
    parser.add_argument("--split", type=str, default="demo")
    parser.add_argument("--model", choices=["gpt35", "gpt4", "llama"], default="llama")
    parser.add_argument("--workers", type=int, default=10, help="parallel processors")
    parser.add_argument("--starting", type=int, default=0)
    parser.add_argument("--ending", type=int, default=None)
    args = parser.parse_args()
    return args


def main(opts: argparse.Namespace):
    model = MODEL_DICT[opts.model]
    span_labeler = SpanLabeler(model, opts.dataset, opts.split)
    save_path = os.path.join(
        SYNTHESIZED_SPAN_LABELING_DATA_PATH, opts.dataset, f"{opts.split}.jsonl"
    )
    with open(save_path, "w+", encoding="utf-8") as f:
        for labeled in span_labeler.parse(workers=opts.workers):
            info = json.dumps(labeled, ensure_ascii=False)
            f.write(info + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
