import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import (
    NEGATIVE_TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT,
    TOKEN_LABELING_SYSTEM_MSG,
)
from tqdm.rich import tqdm_rich

from conf import (
    MODEL_DICT,
    SYNTHESIZED_NEGATIVE_SAMPLING_DATA_PATH,
    SYNTHESIZED_NEGATIVE_SAMPLING_LABELED_DATA_PATH,
)
from language_models import AOAI
from utils import ask_model, load_jsonl


class NegativeTokenLabeler:
    def __init__(self, model: str, dataset: str, split: str) -> None:
        # self.model = AOAI(model)
        negative_sampling_data = os.path.join(SYNTHESIZED_NEGATIVE_SAMPLING_DATA_PATH, dataset, f"{split}.jsonl")
        self.negative_sampling_data = load_jsonl(negative_sampling_data)
        self.check_if_valid = lambda x: all([k in x.keys() for k in ["extracted_words"]])

    def parse(self, starting: int = 0, ending: int = None, workers=10) -> list[dict]:
        if ending is None:
            ending = len(self.negative_sampling_data)
        labeled_data = self.negative_sampling_data[starting:ending]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tasks = {executor.submit(self.parse_sample, sample): idx for idx, sample in enumerate(labeled_data)}
                results = []
                for future in tqdm_rich(as_completed(tasks), total=len(tasks), desc="Processing..."):
                    task_id = tasks[future]
                    try:
                        result = future.result()
                        results.append((task_id, result))
                    finally:
                        pass
            results = [r[1] for r in sorted(results, key=lambda x: x[0])]
        else:
            results = []
            for idx, sample in tqdm_rich(enumerate(labeled_data), total=len(labeled_data), desc="Processing..."):
                result = self.parse_sample(sample)
                results.append(result)
        return results

    def parse_sample(self, sample: dict) -> dict:
        for subq_id, subq in sample["decomposed_questions"].items():
            # prompt = NEGATIVE_TOKEN_LABEL_SYNTHESIZE_FEW_SHOT_PROMPT.format(
            #     question=subq["sub_question"], paragraph=subq["negative_paragraph"]
            # )
            # result = ask_model( self.model, prompt, TOKEN_LABELING_SYSTEM_MSG, type="json", check_if_valid=self.check_if_valid,)
            result = {"extracted_words": ""}
            # if result is None:
            #     sample["decomposed_questions"][subq_id]["state"] = "error"
            #     continue
            sample["decomposed_questions"][subq_id]["negative_extracted_words"] = result["extracted_words"]
        return sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hotpotQA", "musique", "2WikiMQA"],
    )
    parser.add_argument("--split", type=str, required=True, default="demo")
    parser.add_argument("--model", type=str, default="gpt35")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    return args


def main(opt: argparse.Namespace):
    model = MODEL_DICT[opt.model]
    labeler = NegativeTokenLabeler(model, opt.dataset, opt.split)
    with open(
        os.path.join(
            SYNTHESIZED_NEGATIVE_SAMPLING_LABELED_DATA_PATH,
            opt.dataset,
            f"{opt.split}.jsonl",
        ),
        "w",
    ) as f:
        for sample in labeler.parse():
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
