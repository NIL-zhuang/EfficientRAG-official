import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from prompts import (
    TOKEN_LABEL_REDUNDANT_EVALUATION_PROMPT,
    TOKEN_LABEL_REDUNDANT_SYSTEM_MSG,
    TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_MUSIQUE,
    TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_WIKIMQA,
    TOKEN_LABELING_SYSTEM_MSG,
)

from conf import (
    MODEL_DICT,
    SYNTHESIZED_DECOMPOSED_DATA_PATH,
    SYNTHESIZED_TOKEN_LABELING_DATA_PATH,
)
from language_models import LanguageModel, get_model
from utils import ask_model, ask_model_in_parallel, load_jsonl
from utils.model import get_type_parser

TOKEN_LABEL_PROMPT_TEMPLATE_MAPPING = {
    "musique": TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_MUSIQUE,
    "musique-simple": TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_MUSIQUE,
    "2WikiMQA": TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_WIKIMQA,
}


class TokenLabeler:
    def __init__(self, model: str, dataset: str, split: str) -> None:
        self.model: LanguageModel
        self.model = get_model(model)

        labeled_data_path = os.path.join(
            SYNTHESIZED_DECOMPOSED_DATA_PATH, dataset, f"{split}.jsonl"
        )
        self.labeled_data = load_jsonl(labeled_data_path)
        self.check_if_valid = lambda x: all(
            [k in x.keys() for k in ["extracted_words"]]
        )
        self.token_labeling_prompt = TOKEN_LABEL_PROMPT_TEMPLATE_MAPPING[dataset]

    def parse(self, starting: int = 0, workers=10) -> list[dict]:
        labeled_data = self.labeled_data[starting:]
        prompts = []

        labeled_data = [d for d in labeled_data if d.get("state", None) is None]
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {
                executor.submit(self.parse_sample, sample): idx
                for idx, sample in enumerate(labeled_data)
            }
            for task in tqdm_rich(
                as_completed(tasks), total=len(tasks), desc="Processing..."
            ):
                idx = tasks[task]
                try:
                    result = task.result()
                finally:
                    ...
                results.append((result, idx))
        results = [r[0] for r in sorted(results, key=lambda x: x[1])]
        return results

    def parse_sample(self, sample: dict) -> dict:
        prompt_list = self.parse_prompt(sample)
        results = []
        for prompt in prompt_list:
            result = ask_model(
                self.model,
                prompt,
                TOKEN_LABELING_SYSTEM_MSG,
                type="json",
                check_if_valid=self.check_if_valid,
            )
            if result is None:
                result = {"extracted_words": "", "status": "error"}
            results.append(result)
        for subq_id, result in zip(
            sorted(sample["decomposed_questions"].keys()), results
        ):
            chunk = sample["decomposed_questions"][subq_id]
            chunk["extracted_words"] = result["extracted_words"]
        return sample

    def parse_prompt(self, data: dict) -> list[dict]:
        prompt_list = []
        for subq_id in sorted(data["decomposed_questions"].keys()):
            subq = data["decomposed_questions"][subq_id]
            format_kwargs = {
                "question": subq["sub_question"],
                "paragraph": subq["positive_paragraph"],
                "answer": subq["answer"],
            }
            prompt = self.token_labeling_prompt.format(**format_kwargs)
            prompt_list.append(prompt)
        return prompt_list

    def parse_failed(self, token_labeled_data: list[dict]) -> list[dict]:
        results = []
        failed_question_ids = set()
        for sample in token_labeled_data:
            for sub_question_id in sorted(sample["decomposed_questions"].keys()):
                if (
                    sample["decomposed_questions"][sub_question_id].get("state", None)
                    == "error"
                ):
                    failed_question_ids.add(sample["id"])
                    break
        progress = tqdm_rich(
            desc="Processing failed...", total=len(failed_question_ids)
        )
        for sample in token_labeled_data:
            if sample["id"] not in failed_question_ids:
                results.append(sample)
                continue
            for sub_question_id in sorted(sample["decomposed_questions"].keys()):
                if (
                    sample["decomposed_questions"][sub_question_id].get("state", None)
                    != "error"
                ):
                    continue
                prompt_list = self.parse_prompt(sample)
                prompt = prompt_list[int(sub_question_id) - 1]
                result = ask_model(
                    self.model, prompt, type="json", check_if_valid=self.check_if_valid
                )
                if result is None:
                    continue
                del sample["decomposed_questions"][sub_question_id]["state"]
                sample["decomposed_questions"][sub_question_id]["extracted_words"] = (
                    result["extracted_words"]
                )
            progress.update(1)
            results.append(sample)
        return results


class TokenReLabeler:
    def __init__(self, model: str, dataset: str, split: str) -> None:
        self.model: LanguageModel
        self.model = get_model(model)
        self.model_powerful = get_model("Llama3-8B-Instruct")

        labeled_data_path = os.path.join(
            SYNTHESIZED_TOKEN_LABELING_DATA_PATH, dataset, f"{split}.jsonl"
        )
        self.labeled_data = load_jsonl(labeled_data_path)
        self.check_if_valid = lambda x: all(
            [k in x.keys() for k in ["extracted_words"]]
        )
        self.check_redundant_valid = lambda x: type(x) == dict and all(
            [k in x.keys() for k in ["redundant", "missing"]]
        )
        self.type_parser = get_type_parser(type="json")

    def label_redundant(self, labeled_data: list[dict], workers: int) -> list[dict]:
        redundant_questions = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {
                executor.submit(self.check_sample_redundant, sample): idx
                for idx, sample in enumerate(labeled_data)
            }
            for future in tqdm_rich(
                as_completed(tasks), total=len(tasks), desc="Redundant"
            ):
                task_id = tasks[future]
                try:
                    result = future.result()
                    redundant_questions.append((task_id, result))
                finally:
                    ...
            redundant_questions = [
                r[1] for r in sorted(redundant_questions, key=lambda x: x[0])
            ]
        for sample, redundant in zip(labeled_data, redundant_questions):
            assert redundant["id"] == sample["id"]
            for subq_id in redundant["redundant"]:
                sample["decomposed_questions"][subq_id]["redundant"] = True
        return labeled_data

    def parse(self, workers: int = 10, redundant_labeled: bool = False) -> list[dict]:
        labeled_data = [
            d
            for d in self.labeled_data
            if all(
                chunk.get("state", None) is None
                for chunk in d["decomposed_questions"].values()
            )
        ]

        # 1. use GPT3.5 to identify if extracted words is redundant or missing
        if not redundant_labeled:
            labeled_data = self.label_redundant(labeled_data, workers)

        # 2. use Llama3 to re-label the extracted words
        results = []
        data_mapping = {d["id"]: d for d in labeled_data}

        max_iter = 5
        current_iter = 0
        while current_iter < max_iter:
            current_iter += 1

            prompts = []
            for sample in labeled_data:
                sample_prompts = self.build_relabel_prompt(sample)
                prompts.extend(sample_prompts)

            print(
                f"Current iteration: {current_iter}, "
                f"max iteration: {max_iter}, "
                f"handling {len(prompts)} prompts."
            )
            if len(prompts) <= 0:
                break

            batched_prompts = [p["prompt"] for p in prompts]
            results = self.model_powerful.chat(
                batched_prompts, TOKEN_LABELING_SYSTEM_MSG, json_mode=True
            )

            for prompt, result in zip(prompts, results):
                try:
                    json_result = self.type_parser(result)
                    data = data_mapping[prompt["id"]]
                    chunk = data["decomposed_questions"][prompt["subq_id"]]
                    chunk["extracted_words_old"] = chunk["extracted_words"]
                    chunk["extracted_words"] = json_result["extracted_words"]
                    chunk["redundant"] = False
                except json.JSONDecodeError:
                    print(f"Error on {prompt['id']} sub-question {prompt['subq_id']}")
                    json_result = None

        return labeled_data

    def build_relabel_prompt(self, sample: dict):
        prompts = []
        for subq_id, chunk in sample["decomposed_questions"].items():
            if not chunk.get("redundant", False):
                break
            sub_question = chunk["sub_question"]
            paragraph = chunk["positive_paragraph"]
            answer = chunk["answer"]
            prompt = TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT_MUSIQUE.format(
                question=sub_question, paragraph=paragraph, answer=answer
            )
            info = {
                "id": sample["id"],
                "subq_id": subq_id,
                "prompt": prompt,
            }
            prompts.append(info)
        return prompts

    def check_sample_redundant(self, sample: dict):
        redundant = {"id": sample["id"], "redundant": []}
        for subq_id, subq in sample["decomposed_questions"].items():
            question = subq["sub_question"]
            answer = subq["answer"]
            extracted_words = subq["extracted_words"]
            evaluation_prompt = TOKEN_LABEL_REDUNDANT_EVALUATION_PROMPT.format(
                question=question, answer=answer, extracted_words=extracted_words
            )
            evaluation = ask_model(
                self.model,
                evaluation_prompt,
                TOKEN_LABEL_REDUNDANT_SYSTEM_MSG,
                type="json",
                check_if_valid=self.check_redundant_valid,
            )
            if evaluation["redundant"] or evaluation["missing"]:
                redundant["redundant"].append(subq_id)
        return redundant


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="musique",
    )
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--model", default="gpt4")
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel processors"
    )
    parser.add_argument(
        "--sync", action="store_true", help="Syncing with label fixed data"
    )
    parser.add_argument("--failed", action="store_true", help="Parse failed data")
    parser.add_argument("--failed_path", type=str, help="Path to failed data")
    parser.add_argument(
        "--relabel", action="store_true", help="Re-label extracted words"
    )
    args = parser.parse_args()
    return args


def main(opt: argparse.Namespace):
    model = MODEL_DICT[opt.model]
    labeler = TokenLabeler(model, opt.dataset, opt.split)
    with open(
        os.path.join(
            SYNTHESIZED_TOKEN_LABELING_DATA_PATH,
            opt.dataset,
            f"{opt.split}.jsonl",
        ),
        "w+",
        encoding="utf-8",
    ) as f:
        for labeled in labeler.parse(workers=opt.workers):
            info = json.dumps(labeled, ensure_ascii=False)
            f.write(info + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
