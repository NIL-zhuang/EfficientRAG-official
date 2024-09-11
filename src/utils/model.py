import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal

from tenacity import retry, stop_after_attempt
from tqdm.rich import tqdm_rich

from language_models import LanguageModel


@retry(stop=stop_after_attempt(3), reraise=False, retry_error_callback=lambda x: None)
def ask_model(
    model: LanguageModel,
    prompt: str,
    system_msg: str = None,
    type: Literal["json", "text"] = "json",
    check_if_valid: Callable = None,
    sleep: bool = True,
    mode: Literal["chat", "completion"] = "chat",
) -> dict:
    if sleep:
        sleep_time = random.uniform(1.0, 3.0)
        time.sleep(sleep_time)
    if mode == "chat":
        result = model.chat(prompt, system_msg, json_mode=(type == "json"))
        # print(result)
    elif mode == "completion":
        result = model.complete(prompt)
    parser = get_type_parser(type)
    info = parser(result)
    if check_if_valid is not None and not check_if_valid(info):
        print(f"Invalid response {info}")
        raise ValueError("Invalid response")
    return info


def ask_model_in_parallel(
    model: LanguageModel,
    prompts: list[str],
    system_msg: str = None,
    type: Literal["json", "text"] = "json",
    check_if_valid_list: list[Callable] = None,
    max_workers: int = 4,
    desc: str = "Processing...",
    verbose=True,
    mode: Literal["chat", "completion"] = "chat",
):
    if max_workers == -1:
        max_workers = len(prompts)
    assert max_workers >= 1, "max_workers should be greater than or equal to 1"
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if check_if_valid_list is None:
            check_if_valid_list = [None] * len(prompts)
        assert len(prompts) == len(
            check_if_valid_list
        ), "Length of prompts and check_if_valid_list should be the same"
        tasks = {
            executor.submit(
                ask_model, model, prompt, system_msg, type, check_if_valid, mode
            ): idx
            for idx, (prompt, check_if_valid) in enumerate(
                zip(prompts, check_if_valid_list)
            )
        }
        results = []
        for future in tqdm_rich(
            as_completed(tasks), total=len(tasks), desc=desc, disable=not verbose
        ):
            task_id = tasks[future]
            try:
                result = future.result()
                results.append((task_id, result))
            finally:
                ...
        results = [result[1] for result in sorted(results, key=lambda r: r[0])]
        return results


def get_type_parser(type: str) -> Callable:
    def json_parser(result: str):
        # pattern = r"```json(.*?)```"
        pattern = r"{.*?}"
        matches = re.findall(pattern, result, re.DOTALL)
        if matches:
            result = matches[0].strip()
        return json.loads(result)

    def text_parser(result: str):
        return result

    if type == "json":
        return json_parser
    elif type == "text":
        return text_parser
    else:
        raise ValueError(f"Unsupported type: {type}")
