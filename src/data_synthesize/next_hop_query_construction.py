import argparse
import json
import os
import sys
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts.query_labeling import *
from tqdm.rich import tqdm_rich

from conf import (
    MODEL_DICT,
    SYNTHESIZED_NEXT_QUERY_DATA_PATH,
    SYNTHESIZED_TOKEN_EXTRACTED_DATA_PATH,
)
from language_models import LanguageModel, get_model
from utils import ask_model, load_jsonl


class NextQueryFilter:
    def __init__(self, model: str, dataset: str, split: str) -> None:
        self.model: LanguageModel
        self.model = get_model(model)
        self.dataset = dataset

        tagged_data_path = os.path.join(
            SYNTHESIZED_TOKEN_EXTRACTED_DATA_PATH, dataset, f"{split}.jsonl"
        )
        self.tagged_data = load_jsonl(tagged_data_path)
        self.check_if_valid = lambda x: all([k in x.keys() for k in ["filtered_query"]])

    def parse(
        self,
        workers: int = 10,
        hierarchy: list[str] = None,
    ):
        labeled_data = [
            d
            for d in self.tagged_data
            if all(
                d["decomposed_questions"][sub_id].get("state", None) is None
                for sub_id in d["decomposed_questions"].keys()
            )
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {
                executor.submit(self.parse_sample, sample, hierarchy): idx
                for idx, sample in enumerate(labeled_data)
            }
            results = []
            for future in tqdm_rich(
                as_completed(tasks), total=len(tasks), desc="Processing..."
            ):
                task_id = tasks[future]
                try:
                    result = future.result()
                    results.append((task_id, result))
                except Exception as e:
                    print(f"Failed at sample {task_id}: {e}")
        results = [result[1] for result in sorted(results, key=lambda x: x[0])]
        return results

    def parse_sample(
        self,
        sample: dict,
        hierarchy: Optional[tuple[str]] = None,
        larger_model: Optional[LanguageModel] = None,
    ) -> dict:
        if hierarchy is not None and sample["id"].split("_")[0] in hierarchy:
            assert larger_model is not None
            model = larger_model
        else:
            model = self.model
        for subq_id, subq in sample["decomposed_questions"].items():
            if len(subq["dependency"]) == 0:
                subq["filtered_query"] = sample["question"]

        max_iter = 5
        cur_iter = 0
        while cur_iter < max_iter:
            cur_iter += 1
            prompt_list, cur_sub_question_list = self.parse_prompt(sample)
            if len(prompt_list) == 0:
                break
            # print('\n\n'.join(prompt_list) + '\n\n')
            for prompt, subq_id in zip(prompt_list, cur_sub_question_list):
                result = ask_model(
                    model,
                    prompt,
                    QUERY_LABEL_SYSTEM_PROMPT,
                    type="json",
                    check_if_valid=self.check_if_valid,
                )
                if result is None:
                    sample["decomposed_questions"][subq_id][
                        "filtered_query_state"
                    ] = "error"
                    print(f"Error in {sample['id']}: {subq_id}")
                    continue
                sample["decomposed_questions"][subq_id]["filtered_query"] = result[
                    "filtered_query"
                ]
        return sample

    def build_already_known(self, infos: list[str]) -> str:
        return "\n".join([INFO_TEMPLATE.format(info=info) for info in infos])

    def build_sub_answer_list(self, sub_answers: list[str]) -> str:
        return "\n".join(
            [
                SUB_ANSWER_TEMPLATE.format(sub_answer=sub_answer)
                for sub_answer in sub_answers
            ]
        )

    def parse_prompt(self, data: dict) -> list[dict]:
        prompt_list = []
        cur_sub_question_ids = []
        for sub_question_id in sorted(data["decomposed_questions"].keys()):
            # identify which sub_question can generate filtered_question
            chunk = data["decomposed_questions"][sub_question_id]
            if "filtered_query" in chunk.keys():
                # remove chunk that already has filtered_question
                continue
            dependency = chunk["dependency"].copy()
            for dependent_id in dependency:
                # remove duplicated and false question
                if (
                    dependent_id not in data["decomposed_questions"].keys()
                    or dependent_id == sub_question_id
                ):
                    dependency.remove(dependent_id)
            if not all(
                [
                    "filtered_query" in data["decomposed_questions"][dep_id].keys()
                    for dep_id in dependency
                ]
            ):
                # remove questions that cannot generate filtered_question in this iter
                continue

            # label sub-question with few-shot prompt of different situations
            info_list = [
                data["decomposed_questions"][dep_id]["matched"] for dep_id in dependency
            ]

            if self.dataset == "2WikiMQA" and len(data["decomposed_questions"]) == 4:
                # 2WikiMQA Bridge-Comparison case
                info_list = [
                    subq["matched"]
                    for subq in data["decomposed_questions"].values()
                    if len(subq["dependency"]) == 0
                ]

            sub_answer_list = [
                data["decomposed_questions"][dep_id]["answer"] for dep_id in dependency
            ]
            info_list = self.build_already_known(info_list)
            sub_answers = self.build_sub_answer_list(sub_answer_list)
            prev_question = data["decomposed_questions"][dependency[0]][
                "filtered_query"
            ]
            prompt_template = self.build_prompt_template(data, dependency)
            prompt = prompt_template.format(
                question=prev_question, info_list=info_list, subq_answers=sub_answers
            )
            prompt_list.append(prompt)
            cur_sub_question_ids.append(sub_question_id)
        return prompt_list, cur_sub_question_ids

    def build_prompt_template(self, sample: dict, dependency: list[str]) -> str:
        build_prompt_template_mapping = {
            "hotpotQA": self.build_prompt_template_hotpot,
            "2WikiMQA": self.build_prompt_template_2wiki,
            "musique": self.build_prompt_template_musique,
        }
        build_prompt_template_func = build_prompt_template_mapping[self.dataset]
        return build_prompt_template_func(sample, dependency)

    def build_prompt_template_hotpot(self, sample: dict, dependency: list[str]) -> str:
        raise NotImplementedError()

    def build_prompt_template_2wiki(self, sample: dict, dependency: list[str]) -> str:
        # Comparison type as already been done in the previous step
        prompt_template = None
        if len(sample["decomposed_questions"]) == 2:
            # Q -> A -> B
            # inference and compositional
            prompt_template = QUERY_LABEL_COMPOSITIONAL_2WIKIMQA
        elif len(sample["decomposed_questions"]) == 4:
            # Q -> A, B -> C, D
            # bridge_comparison
            prompt_template = QUERY_LABEL_BRIDGE_COMPARISON_2WIKIMQA
        else:
            raise NotImplementedError
        return prompt_template

    def build_prompt_template_musique(self, sample: dict, dependency: list[str]) -> str:
        prompt_template = None
        if (
            sample["decomposed_questions"][dependency[0]]["filtered_query"]
            == sample["question"]
            and len(dependency) == 1
        ):
            # dependent prev-question is the original question
            # subq-B ==>> Q -> A -> B
            prompt_template = QUERY_LABEL_AFTER_ORIGINAL_QUESTION_MUSIQUE
        elif len(dependency) == 1 and any(
            sample["decomposed_questions"][dep_id]["dependency"] != []
            for dep_id in dependency
        ):
            # dependent on 1 sub-question & is filtered_query
            # subq-C ==>> Q -> A -> B -> C
            # subq-D ==>> Q -> A -> B -> C -> D
            prompt_template = QUERY_LABEL_AFTER_FILTERED_QUESTION_MUSIQUE
        elif len(dependency) > 1:
            # subq-C ==>> Q -> (A, B) -> C
            # subq-D ==>> Q -> A -> B, Q -> C, (B, C) -> D
            prompt_template = QUERY_LABEL_FROM_MULTI_SOURCE_MUSIQUE
        else:
            raise NotImplementedError
        return prompt_template

    def parse_failed(self, query_filtered_data: list[dict]) -> list[dict]:
        # TODO its just copied but not implemented
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA", "2WikiMQA-small"],
        default="musique",
    )
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--model", default="llama")
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel processors"
    )
    args = parser.parse_args()
    return args


def main(opt: argparse.Namespace):
    filter = NextQueryFilter(opt.model, opt.dataset, opt.split)

    output_path = os.path.join(
        SYNTHESIZED_NEXT_QUERY_DATA_PATH, opt.dataset, f"{opt.split}.jsonl"
    )
    with open(output_path, "w+", encoding="utf-8") as f:
        filtered_query_list = filter.parse(workers=opt.workers)
        for filtered_sample in filtered_query_list:
            info = json.dumps(filtered_sample, ensure_ascii=False)
            f.write(info + "\n")


if __name__ == "__main__":
    options = parse_args()
    main(options)
