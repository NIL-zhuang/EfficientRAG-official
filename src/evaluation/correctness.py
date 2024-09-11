import argparse
import os
import re
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from language_models import LanguageModel, get_model
from utils import ask_model, load_jsonl, write_jsonl

LLM_EVAL_PROMPT = """
You are an experienced linguist who is responsible for evaluating the correctness of the generated responses.
You are provided with question, the generated responses and the corresponding ground truth answer.
Your task is to compare the generated responses with the ground truth responses and evaluate the correctness of the generated responses.
Response in JSON format with key "response" and value "yes" or "no".

Question: {question}
Prediction: {prediction}
Ground-truth Answer: {answer}
Your response:
""".strip()

LLM_EXTRACT_ANSWER_PROMPT = """
Given a question, you should simplify the response to a more concise form of answer. If the response is already in a concise form, you can response with the same answer. If the response does not contain the answer, you can return "noanswer".
You should come out the simplified answer in JSON format with key "answer" and the answer string as the value. Your response should be in markdown code block. Like
```json
{"answer": "simplified answer"}
```

Question: {question}
Response: {response}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, required=True)
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--extract_answer", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    return args


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(ground_truth)
    if normalized_prediction in normalized_answer:
        return 1
    return 0


def acc_evaluate(question: str, answer: str, prediction: str, model: LanguageModel):
    prompt = LLM_EVAL_PROMPT.format(
        question=question, prediction=prediction, answer=answer
    )
    response = ask_model(
        model,
        prompt,
        mode="chat",
        type="json",
        check_if_valid=lambda resp: type(resp) is dict
        and "response" in resp
        and resp["response"] in ["yes", "no"],
    )
    if response is None:
        return False
    return response["response"] == "yes"


def extract_answer(question: str, response: str, model: LanguageModel):
    prompt = LLM_EXTRACT_ANSWER_PROMPT.format(question=question, response=response)
    result = ask_model(
        model,
        prompt,
        mode="chat",
        type="json",
        check_if_valid=lambda resp: type(resp) is dict and "answer" in resp,
    )
    if result is None:
        return response
    return result["answer"]


def evaluate_sample(sample, model: LanguageModel):
    question = sample["question"]
    answer = sample["answer"]

    if type(answer) is list:
        answer = " ".join(answer)
    assert type(answer) is str, f"Answer is not a string: {answer}"  # noqa

    prediction = sample["model_output"]
    # prediction = sample["model_answer"]
    origin_pred = prediction
    if opts.extract_answer:
        prediction = extract_answer(question, prediction, model)

    acc = acc_evaluate(question, answer, prediction, model)
    f1, precision, recall = f1_score(prediction, answer)
    em = exact_match(prediction, answer)
    return {
        "question": question,
        "answer": answer,
        "origin_prediction": origin_pred,
        "prediction": prediction,
        "correctness": acc,
        "f1": f1,
        "em": em,
    }


def main(opts: argparse.Namespace):
    fpath = opts.fpath
    data = load_jsonl(fpath)
    model = get_model(opts.model)
    base_path = os.path.splitext(fpath)[0]
    results = []
    with ThreadPoolExecutor(max_workers=opts.workers) as executor:
        tasks = {
            executor.submit(evaluate_sample, sample, model): idx
            for idx, sample in enumerate(data)
        }

        for future in tqdm_rich(
            as_completed(tasks), total=len(tasks), desc="Evaluating"
        ):
            try:
                idx = tasks[future]
                result = future.result()
                results.append((result, idx))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
            finally:
                pass

    results = [result for result, _ in sorted(results, key=lambda x: x[1])]

    output_path = f"{base_path}_correctness.jsonl"
    write_jsonl(results, output_path)

    accuracy_score = sum([result["correctness"] for result in results]) / len(results)
    f1_score_avg = sum([result["f1"] for result in results]) / len(results)
    em_score_avg = sum([result["em"] for result in results]) / len(results)
    print(f"EM: {em_score_avg:.4f}")
    print(f"F1: {f1_score_avg:.4f}")
    print(f"Accuracy: {accuracy_score:.4f}")


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
