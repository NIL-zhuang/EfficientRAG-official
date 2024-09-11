import argparse
import os
import sys

import numpy as np
from tqdm.rich import tqdm_rich
from vllm import LLM, SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from conf import CORPUS_DATA_PATH, SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH
from retrievers import Retriever
from utils import load_jsonl, write_jsonl

MODEL_CKPT = {
    "7b": "model_cache/selfrag_llama2_7b",
    "13b": "model-cache/selfrag_llama2_13b",
}

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = [
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]
ground_tokens_names = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
]
other_special_tokens = ["<s>", "</s>", "[PAD]", "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = [
    "[Fully supported]",
    "[Partially supported]",
    "[No support / Contradictory]",
    "[No Retrieval]",
    "[Retrieval]",
    "[Irrelevant]",
    "[Relevant]",
    "<paragraph>",
    "</paragraph>",
    "[Utility:1]",
    "[Utility:2]",
    "[Utility:3]",
    "[Utility:4]",
    "[Utility:5]",
]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in retrieval_tokens_names
    }
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA"],
        default="musique",
    )
    parser.add_argument("--split", type=str, default="demo")
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=10)
    args = parser.parse_args()
    return args


class SelfRAG:
    def __init__(
        self,
        model_ckpt: str,
        dataset: list[dict],
        retriever: Retriever,
        use_grounding: bool = False,
        use_utility: bool = False,
    ) -> None:
        self.model = LLM(model_ckpt, dtype="half")
        self.model.llm_engine.model_config.max_logprobs = 32017
        self.tokenizer = self.model.get_tokenizer()
        self.dataset = dataset
        self.retriever = retriever
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = (
            self.load_special_tokens(self.tokenizer, use_grounding, use_utility)
        )

    def inference(self, workers: int = 10) -> list[dict]:
        results = []
        for idx, sample in tqdm_rich(enumerate(self.dataset), total=len(self.dataset)):
            result = self.infer_sample(sample)
            results.append(result)
        return results

    def infer_sample(self, sample: dict) -> dict:
        question = sample["question"]
        results = {
            "id": sample["id"],
            "answer": sample["answer"],
            "oracle": [
                f"{sample['id']}-{'{:02d}'.format(chunk['positive_paragraph_idx'])}"
                for chunk in sample["decomposed_questions"].values()
            ],
            "question": sample["question"],
            "inter": {},
        }
        prompt = self.format_prompt(question)
        evidences = self.retriever.search(question, 10)[0]
        pred, res, do_retrieve = self.generate(
            prompt, evidences, max_new_tokens=15, threshold=0.4
        )
        if do_retrieve:
            results["retrieve"] = True
        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        results["pred"] = pred
        if do_retrieve:
            results["chunks"] = evidences
            best_score = 0
            best_ctx = None
            for k, v in res.items():
                if k == "no_retrieval":
                    continue
                score = v["score"]
                if score > best_score:
                    best_ctx = v["ctx"]
            results["best_ctx"] = best_ctx
        else:
            results["chunks"] = []
            results["best_ctx"] = None
        return results

    def format_prompt(self, input, paragraph=None):
        prompt = f"### Instruction:\n{input}\n\n### Response:\n"
        if paragraph is not None:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt

    def load_special_tokens(self, tokenizer, use_grounding=False, use_utility=False):
        ret_tokens = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in retrieval_tokens_names
        }
        rel_tokens = {}
        for token in ["[Irrelevant]", "[Relevant]"]:
            rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        grd_tokens = None
        if use_grounding is True:
            grd_tokens = {}
            for token in ground_tokens_names:
                grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        ut_tokens = None
        if use_utility is True:
            ut_tokens = {}
            for token in utility_tokens_names:
                ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens

    def generate(
        self,
        prompt: str,
        evidences: list[dict[str]],
        max_new_tokens: int = 15,
        use_seq_score=False,
        threshold: float = 0.5,
        w_rel=1.0,
        w_sup=1.0,
        w_use=0.5,
        mode: str = "adaptive_retrieval",
        closed: bool = False,
    ):
        results = {}
        if mode != "always_retrieve":
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016
            )
            preds = self.model.generate([prompt], sampling_params, use_tqdm=False)
            pred_token_ids = preds[0].outputs[0].token_ids
            pred_text = preds[0].outputs[0].text
            pred_log_probs = preds[0].outputs[0].logprobs
            results["no_retrieval"] = pred_text

        # save relevance token scores
        if mode == "always_retrieve":
            do_retrieve = True

        elif mode == "no_retrieval":
            do_retrieve = False
        else:
            if threshold is not None:
                score_dict = {}
                for tok, id in self.ret_tokens.items():
                    if id not in pred_log_probs[0]:
                        score_dict[tok] = -100
                    prob = pred_log_probs[0][id]
                    score_dict[tok] = prob.logprob
                do_retrieve = (
                    score_dict["[Retrieval]"]
                    / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])
                    > threshold
                )
            else:
                do_retrieve = "[Retrieval]" in pred_token_ids

        if do_retrieve is True:

            def parse_doc(doc: str):
                idx = doc.find(": ")
                if idx == -1:
                    return "", doc

                title = doc[:idx]
                text = doc[idx + 2 :]
                return title, text

            evidence_list = []
            for evidence in evidences:
                doc = evidence["text"]
                title, text = parse_doc(doc)
                info = {"title": title, "text": text}
                evidence_list.append(info)

            evidence_augmented_inputs = [
                prompt + f"[Retrieval]<paragraph>{e['title']}\n{e['text']}</paragraph>"
                for e in evidence_list
            ]
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000
            )
            preds = self.model.generate(
                evidence_augmented_inputs, sampling_params, use_tqdm=False
            )

            relevance_score_dict = {}
            grd_score_dict = {}
            ut_score_dict = {}
            overall_scores = {}
            for p_idx, pred in enumerate(preds):
                pred_token_ids = pred.outputs[0].token_ids
                pred_text = pred.outputs[0].text
                pred_log_probs = pred.outputs[0].logprobs
                seq_score = pred.outputs[0].cumulative_logprob / max(
                    len(pred.outputs[0].token_ids), 1
                )

                relevance_score_dict.setdefault(p_idx, {})
                grd_score_dict.setdefault(p_idx, {})
                ut_score_dict.setdefault(p_idx, {})
                # Compute reward scores
                for tok, id in self.rel_tokens.items():
                    prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                    relevance_score_dict[p_idx][tok] = np.exp(prob.logprob)

                if self.grd_tokens is not None:
                    groundness_token_appear_indices = []
                    for tok_idx, tok in enumerate(pred_token_ids):
                        if tok in list(self.grd_tokens.values()):
                            groundness_token_appear_indices.append(tok_idx)
                            break
                    if len(groundness_token_appear_indices) > 0:
                        idx = groundness_token_appear_indices[0]
                        for token, token_id in self.grd_tokens.items():
                            prob = (
                                pred_log_probs[idx][token_id]
                                if token_id in pred_log_probs[idx]
                                else -100
                            )
                            grd_score_dict[p_idx][token] = np.exp(prob.logprob)

                if self.ut_tokens is not None:
                    utility_token_appear_indices = []
                    for tok_idx, tok in enumerate(pred_token_ids):
                        if tok in list(self.ut_tokens.values()):
                            utility_token_appear_indices.append(tok_idx)
                    if len(utility_token_appear_indices) > 0:
                        idx = utility_token_appear_indices[0]
                        for token, token_id in self.ut_tokens.items():
                            prob = (
                                pred_log_probs[idx][token_id]
                                if token_id in pred_log_probs[idx]
                                else -100
                            )
                            ut_score_dict[p_idx][token] = np.exp(prob.logprob)

                relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                    np.sum(list(relevance_score_dict[p_idx].values()))
                )

                if len(grd_score_dict[p_idx]) == 3:
                    gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                    ground_score = (
                        grd_score_dict[p_idx]["[Fully supported]"] / gt_sum
                    ) + 0.5 * (grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
                else:
                    ground_score = 0.0

                if len(ut_score_dict[p_idx]) == 5:
                    ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                    ut_scores = [-1, -0.5, 0, 0.5, 1]
                    utility_score = np.sum(
                        [
                            ut_scores[i]
                            * (
                                ut_score_dict[p_idx]["[Utility:{}]".format(i + 1)]
                                / ut_sum
                            )
                            for i in range(len(ut_scores))
                        ]
                    )
                else:
                    utility_score = 0.0

                if use_seq_score is True:
                    final_score = (
                        np.exp(seq_score)
                        + w_rel * relevance_score
                        + w_sup * ground_score
                        + w_use * utility_score
                    )
                else:
                    final_score = (
                        w_rel * relevance_score
                        + w_sup * ground_score
                        + w_use * utility_score
                    )

                overall_scores[p_idx] = {
                    "final_score": final_score,
                    "relevance_score": relevance_score,
                    "ground_score": ground_score,
                    "utility_score": utility_score,
                    "relevance_score_dict": relevance_score_dict,
                    "grd_score_dict": grd_score_dict,
                    "ut_score_dict": utility_score,
                }
                results["retrieval_{}".format(p_idx)] = {
                    "pred": pred_text,
                    "score": final_score,
                    "ctx": evidences[p_idx],
                }

        else:
            # No Retrieval
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens
            )
            prompt += "[No Retrieval]"
            preds = self.model.generate([prompt], sampling_params, use_tqdm=False)
            pred = preds[0].outputs[0].text

        # Aggregating answers
        if len(results) == 1:
            postprocessed_pred = self._postprocess_answer_option_conditioned(pred)
            return postprocessed_pred, results, do_retrieve
        else:
            answer2score = {}
            if closed is True:
                for key, result in results.items():
                    if key == "no_retrieval":
                        continue
                    answer = self._postprocess_answer_option_conditioned(result["pred"])
                    score = result["score"]
                    answer2score.setdefault(answer, 0)
                    answer2score[answer] += score
                sorted_answers = sorted(
                    answer2score.items(), key=lambda x: x[1], reverse=True
                )
                best_option = sorted_answers[0][0]
            else:
                path2score = {
                    key: item["score"]
                    for key, item in results.items()
                    if key != "no_retrieval"
                }
                best_path = sorted(
                    path2score.items(), key=lambda x: x[1], reverse=True
                )[0][0]
                best_option = results[best_path]["pred"]
            return best_option, results, do_retrieve

    def _postprocess_answer_option_conditioned(self, answer):
        for token in control_tokens:
            answer = answer.replace(token, "")
        if "</s>" in answer:
            answer = answer.replace("</s>", "")
        if "\n" in answer:
            answer = answer.replace("\n", "")
        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")
        return answer


def main(opts: argparse.Namespace):
    dataset = load_jsonl(
        os.path.join(
            SYNTHESIZED_NEXT_QUERY_EXTRACTED_DATA_PATH,
            opts.dataset,
            f"{opts.split}.jsonl",
        )
    )
    passage_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, "corpus.jsonl")
    embedding_path = os.path.join(CORPUS_DATA_PATH, opts.dataset, opts.retriever)
    retriever = Retriever(
        passage_path=passage_path,
        passage_embedding_path=embedding_path,
        index_path_dir=embedding_path,
        model_type=opts.retriever,
    )
    model_ckpt = MODEL_CKPT[opts.model]
    selfrag = SelfRAG(model_ckpt, dataset, retriever=retriever)
    results = selfrag.inference()
    save_path = os.path.join(
        "results/retrieve/selfrag", f"{opts.dataset}-{opts.split}.jsonl"
    )
    write_jsonl(results, save_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
