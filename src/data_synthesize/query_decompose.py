import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator

from tqdm.rich import tqdm_rich

from conf import MODEL_DICT, SYNTHESIZED_DECOMPOSED_DATA_PATH
from data_module import get_dataset
from data_synthesize.prompts import *
from language_models import LanguageModel, get_model
from utils import ask_model, load_jsonl


class DatasetParser:
    def __init__(self, model: str) -> None:
        self.dataset = None
        self.model: LanguageModel
        self.model = get_model(model)

    def parse(
        self, starting: int = 0, ending: int = None, workers=10
    ) -> Iterator[dict]:
        if ending is None:
            ending = len(self.dataset)
        samples = self.dataset[starting:ending]
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {
                executor.submit(self.process_sample, sample): idx
                for idx, sample in enumerate(samples)
            }
            for task in tqdm_rich(
                as_completed(tasks), total=len(tasks), desc="Processing..."
            ):
                idx = tasks[task]
                try:
                    result = task.result()
                except Exception as e:
                    result = {"state": "failed"}
                results.append((result, idx))
        results = [result for result, idx in sorted(results, key=lambda x: x[1])]
        return results

    def process_sample(self, sample: dict) -> dict:
        prompt = self.parse_sample(sample)
        check_if_valid = self.check_if_valid(sample)
        result = ask_model(
            self.model, prompt, mode="chat", type="json", check_if_valid=check_if_valid
        )
        result = self.post_process(result, sample)
        return result

    def parse_failed(self, data_path: str):
        dataset = load_jsonl(data_path)
        total = len([d for d in dataset if "state" in d.keys()])
        progress = tqdm_rich(dataset, desc="Processing...", total=total)
        results = []
        for sample in dataset:
            if sample.get("state", None) is None:
                results.append(sample)
                continue
            prompt = self.parse_sample(sample)
            result = ask_model(
                self.model,
                prompt,
                type="json",
                check_if_valid=self.check_if_valid(sample),
            )
            results.append(self.post_process(result, sample))
            progress.update(1)
        progress.close()
        with open(os.path.splitext(data_path)[0] + "_fixed.jsonl", "w+") as f:
            for sample in results:
                f.write(json.dumps(sample) + "\n")
        return results

    def parse_sample(self, sample: dict) -> dict:
        raise NotImplementedError()

    def post_process(self, info: dict, sample: dict) -> dict:
        raise NotImplementedError()

    def check_if_valid(self, sample: dict) -> Callable:
        raise NotImplementedError()

    def hierarchical_dataset(self, hard: bool = False):
        return


class WikiMQAParser(DatasetParser):
    def __init__(self, model: str, split: str) -> None:
        super().__init__(model)
        self.prompts = {
            "inference": WikiMQAPromptInference,
            "comparison": WikiMQAPromptComparison,
            "bridge_comparison": WikiMQAPromptBridgeComparison,
            "compositional": WikiMQAPromptCompositional,
        }

        dataset = get_dataset("2WikiMQA", split)
        self.dataset = []
        for sample in dataset:
            if (
                sample["type"] == "bridge_comparison"
                and len(sample["decomposition"]) == 4
            ) or len(sample["decomposition"]) == 2:
                self.dataset.append(sample)
        print(f"Filtered dataset to {len(self.dataset)} samples")

    def parse_sample(self, sample: dict) -> dict:
        question = sample["question"]
        question_type = sample["type"]
        chunks = []
        for idx, item in enumerate(sample["decomposition"]):
            decomposed_str = WikiMQAFactPrompt.format(
                question_id=idx + 1,
                doc_title=item["title"],
                facts=item["chunk"][len(item["title"]) + 1 :].strip(),
                evidence=item["evidence"],
            )
            chunks.append(decomposed_str)
        chunks = "\n\n".join(chunks)
        prompt_template = self.prompts[question_type]
        prompt = prompt_template.format(question=question, chunks=chunks)
        return prompt

    def post_process(self, info: dict, sample: dict) -> dict:
        documents = dict()
        for idx, item in enumerate(sample["decomposition"]):
            documents[f"{idx+1}"] = {
                "title": item["title"],
                "chunk": item["chunk"],
                "evidence": item["evidence"],
                "id": item["id"],
            }
        if info is None:
            sample["state"] = "failed"
            print(f"Failed to synthesize sample {sample['id']}")
            return sample

        for idx, (subq_id, paragraph) in enumerate(
            info["decomposed_questions"].items()
        ):
            doc_id = paragraph["document"]
            paragraph.pop("document")
            paragraph["positive_paragraph"] = documents[doc_id]["chunk"]
            paragraph["evidence"] = documents[doc_id]["evidence"]
            paragraph["positive_paragraph_idx"] = documents[doc_id]["id"]
        info["id"] = sample["id"]
        info["answer"] = sample["answer"]
        return info

    def check_if_valid(self, sample: dict) -> Callable:
        return lambda x: len(x["decomposed_questions"]) == len(
            sample["supporting_facts"]
        )

    def hierarchical_dataset(self, hard: bool = False) -> None:
        hard_question_types = ["bridge_comparison"]
        if hard == True:
            self.dataset = [d for d in self.dataset if d["type"] in hard_question_types]
        else:
            self.dataset = [
                d for d in self.dataset if d["type"] not in hard_question_types
            ]


class HotpotQAParser(DatasetParser):
    def __init__(self, model: str, split: str) -> None:
        super().__init__(model)
        self.dataset = get_dataset("hotpotQA", split)
        self.prompts = {
            "comparison": HotpotQAPromptComparison,
            "compose": HotpotQAPromptCompose,
        }

    def parse_sample(self, sample: dict) -> dict:
        question = sample["question"]
        question_type = sample["type"]
        chunks = []

        for idx, item in enumerate(sample["supporting_facts"]):
            decomposed_str = hotpotQAFactPrompt.format(
                question_id=idx + 1, facts=item["chunk"]
            )
            chunks.append(decomposed_str)
        chunks = "\n".join(chunks)
        prompt_template = self.prompts[question_type]
        if question_type == "comparison":
            prompt = prompt_template.format(question=question, chunks=chunks)
        elif question_type == "compose":
            prompt = prompt_template.format(
                question=question, chunks=chunks, answer=sample["answer"]
            )
        return prompt

    def post_process(self, info: dict, sample: dict) -> dict:
        if info is None:
            sample["state"] = "failed"
            print(f"Failed to synthesize sample {sample['id']}")
            return sample
        for idx, paragraph in enumerate(sample["supporting_facts"]):
            info["decomposed_questions"][f"{idx + 1}"]["positive_paragraph"] = (
                paragraph["chunk"]
            )
            info["decomposed_questions"][f"{idx + 1}"]["positive_paragraph_idx"] = (
                paragraph["id"]
            )
        info["id"] = sample["id"]
        info["answer"] = sample["answer"]
        return info

    def check_if_valid(self, sample: dict) -> Callable:
        return lambda x: len(x["decomposed_questions"]) == len(
            sample["supporting_facts"]
        )

    def hierarchical_dataset(self, hard: bool = False):
        raise NotImplementedError("HotpotQA does not have different difficulty levels")


class MuSiQueParser(DatasetParser):
    def __init__(self, model: str, split: str) -> None:
        super().__init__(model)
        self.dataset = get_dataset("musique-simple", split)
        self.prompt_template_mapping = {
            "2hop": MuSiQueCompose2HopPrompt,
            "3hop1": MuSiQueCompose3HopPrompt,
            "3hop2": MuSiQue3HopSeparateComposePrompt,
            "4hop1": MuSiQueCompose4HopPrompt,
            "4hop2": MuSiQue4HopComposeBridgePrompt,
            "4hop3": MuSiQueAsymmetryBridgePrompt,
        }

    def parse_sample(self, sample: dict) -> dict:
        question = sample["question"]
        decomposed_questions = []
        for idx, (paragraph_idx, decompose_question, decompose_answer) in enumerate(
            zip(
                sample["decomposition"]["paragraph_support_idx"],
                sample["decomposition"]["question"],
                sample["decomposition"]["answer"],
            )
        ):
            paragraph = sample["chunks"][paragraph_idx]
            decomposed_str = MuSiQueSupportingFactPrompt.format(
                question_id=idx + 1,
                sub_question=decompose_question,
                sub_answer=decompose_answer,
            )
            decomposed_questions.append(decomposed_str)
        decomposed_questions = "\n".join(decomposed_questions)
        prompt_template = self.prompt_template_mapping[sample["id"].split("_")[0]]
        prompt = prompt_template.format(
            question=question, decomposed_questions=decomposed_questions
        )
        return prompt

    def post_process(self, info: dict, sample: dict) -> dict:
        if info is None:
            sample["state"] = "failed"
            print(f"Failed to synthesize sample {sample['id']}")
            return sample
        for idx, paragraph_idx in enumerate(
            sample["decomposition"]["paragraph_support_idx"]
        ):
            paragraph = sample["chunks"][paragraph_idx]
            info["decomposed_questions"][f"{idx + 1}"]["positive_paragraph"] = paragraph
            info["decomposed_questions"][f"{idx + 1}"][
                "positive_paragraph_idx"
            ] = paragraph_idx
        info["id"] = sample["id"]
        info["answer"] = sample["answer"]
        return info

    def check_if_valid(self, sample: dict):
        def is_valid(info: dict):
            if len(info["decomposed_questions"]) != len(
                sample["decomposition"]["question"]
            ):
                return False
            return True

        return is_valid

    def hierarchical_dataset(self, hard: bool = False) -> None:
        hard_prefix_set = ["3hop2", "4hop1", "4hop2", "4hop3"]
        if hard == True:
            self.dataset = [
                d for d in self.dataset if d["id"].split("_")[0] in hard_prefix_set
            ]
        else:
            self.dataset = [
                d for d in self.dataset if d["id"].split("_")[0] not in hard_prefix_set
            ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["musique", "musique-simple", "2WikiMQA", "2WikiMQA-small", "hotpotQA"],
        required=True,
    )
    parser.add_argument("--split", type=str, default="demo")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--starting", type=int, default=0)
    parser.add_argument("--ending", type=int, default=None)
    args = parser.parse_args()
    return args


def get_parser(dataset: str, model: str, split: str) -> DatasetParser:
    if dataset in ("musique", "musique-simple"):
        return MuSiQueParser(model, split)
    elif dataset == "hotpotQA":
        return HotpotQAParser(model, split)
    elif dataset == "2WikiMQA":
        return WikiMQAParser(model, split)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented")


def main(opt: argparse.Namespace):
    parser = get_parser(opt.dataset, opt.model, opt.split)

    output_dir = os.path.join(SYNTHESIZED_DECOMPOSED_DATA_PATH, opt.dataset)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{opt.split}.jsonl"), "w+") as f:
        for info in parser.parse(
            workers=opt.workers, starting=opt.starting, ending=opt.ending
        ):
            data = json.dumps(info)
            f.write(data + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
