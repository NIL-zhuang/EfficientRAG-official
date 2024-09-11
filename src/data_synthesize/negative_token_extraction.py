# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
import sys

import spacy
from tqdm.rich import tqdm_rich

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from conf import (
    SYNTHESIZED_NEGATIVE_SAMPLING_EXTRACTED_DATA_PATH,
    SYNTHESIZED_NEGATIVE_SAMPLING_LABELED_DATA_PATH,
)
from utils import load_jsonl, write_jsonl

nlp = spacy.load("en_core_web_sm")


def split_string(input_string, ignore_tokens=set([","])):
    doc = nlp(input_string)
    lemma_word_list = []
    word_list = []
    for word in doc:
        if word.lemma_ not in ignore_tokens:
            lemma_word_list.append(word.lemma_)
            word_list.append(word.text)
    return lemma_word_list, word_list


def is_equal(token1, token2):
    return token1.lower() == token2.lower()


def label_word(
    origin_paragraph: str,
    extracted_words: str,
    window_size: int = 150,
    verbose: bool = False,
):
    lemma_paragraph_tokens, paragraph_tokens = split_string(origin_paragraph)
    lemma_comp_tokens, comp_tokens = split_string(extracted_words)
    origin_lemma_tokens_set = set(lemma_paragraph_tokens)
    for lemma_paragraph_token in lemma_paragraph_tokens:
        origin_lemma_tokens_set.add(lemma_paragraph_token.lower())

    num_find = 0
    prev_idx = 0
    num_origin_tokens = len(lemma_paragraph_tokens)
    labels = [False] * num_origin_tokens
    for lemma_comp_token, comp_token in zip(lemma_comp_tokens, comp_tokens):
        if (
            lemma_comp_token in origin_lemma_tokens_set
            or lemma_comp_token.lower() in origin_lemma_tokens_set
        ):
            num_find += 1
        for i in range(window_size):
            # look forward
            token_idx = min(prev_idx + i, num_origin_tokens - 1)
            if (
                is_equal(lemma_paragraph_tokens[token_idx], lemma_comp_token)
                and not labels[token_idx]
            ):
                labels[token_idx] = True
                # window do not go too fast
                if token_idx - prev_idx > window_size // 2:
                    prev_idx += window_size // 2
                else:
                    prev_idx = token_idx
                if verbose:
                    print(
                        lemma_comp_token,
                        token_idx,
                        prev_idx,
                        lemma_paragraph_tokens[token_idx - 1 : token_idx + 2],
                    )
                break
            # look backward
            token_idx = max(prev_idx - i, 0)
            if (
                is_equal(lemma_paragraph_tokens[token_idx], lemma_comp_token)
                and not labels[token_idx]
            ):
                labels[token_idx] = True
                prev_idx = token_idx
                if verbose:
                    print(
                        lemma_comp_token,
                        token_idx,
                        prev_idx,
                        lemma_paragraph_tokens[token_idx - 1 : token_idx + 2],
                    )
                break

    retrieval_tokens = []
    for idx, token in enumerate(paragraph_tokens):
        if labels[idx]:
            retrieval_tokens.append(token)
    matched = " ".join(retrieval_tokens)

    comp_rate = len(lemma_comp_tokens) / len(lemma_paragraph_tokens)
    if len(lemma_comp_tokens) > 0:
        find_rate = num_find / len(lemma_comp_tokens)
    else:
        find_rate = 0.0
    variation_rate = 1 - find_rate
    hitting_rate = num_find / len(lemma_paragraph_tokens)
    matching_rate = sum(labels) / len(labels)
    alignment_gap = hitting_rate - matching_rate

    if alignment_gap > 0.1:
        print(origin_paragraph)
        print("-" * 50)
        print(extracted_words)
        print("-" * 50)
        print(matched)
        print("-" * 50)
        print(lemma_paragraph_tokens)
        print("-" * 50)
        print(lemma_comp_tokens)
        print("-" * 50)
        print(retrieval_tokens)
        print("=" * 50)
        print(
            f"compress rate: {comp_rate}, variation_rate: {variation_rate}, alignment_gap: {alignment_gap}"
        )

    return {
        "labels": labels,
        "matched": matched,
        "paragraph_tokens": paragraph_tokens,
        "comp_rate": comp_rate,
        "find_rate": find_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap,
    }


def main(opts: argparse.Namespace):
    data = load_jsonl(opts.data_path)

    infos = {
        "comp_rate": 0.0,
        "variation_rate": 0.0,
        "hitting_rate": 0.0,
        "matching_rate": 0.0,
        "alignment_gap": 0.0,
        "find_rate": 0.0,
    }

    num_samples = 0
    for sample in tqdm_rich(data):
        flag = True
        for sid, subq in sample["decomposed_questions"].items():
            if subq.get("negative_extracted_words", None) is None:
                flag = False
                break
        if not flag:
            del sample
            continue
        for sid, subq in sample["decomposed_questions"].items():
            results = label_word(
                subq["negative_paragraph"],
                subq["negative_extracted_words"],
                verbose=opts.verbose,
            )
            subq["negative_labels"] = results["labels"]
            subq["negative_matched"] = results["matched"]
            subq["negative_paragraph_tokens"] = results["paragraph_tokens"]

            num_samples += 1
            for k in infos.keys():
                infos[k] += results[k]

    for k, v in infos.items():
        v = v / num_samples * 100
        print(f"{k}: {v:.2f}")

    os.makedirs(os.path.dirname(opts.save_path), exist_ok=True)
    write_jsonl(data, opts.save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="annotate token")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["hotpotQA", "musique", "2WikiMQA"],
    )
    parser.add_argument("--split", required=True, type=str, default="demo")
    args = parser.parse_args()
    args.data_path = os.path.join(
        SYNTHESIZED_NEGATIVE_SAMPLING_LABELED_DATA_PATH,
        args.dataset,
        f"{args.split}.jsonl",
    )
    args.save_path = os.path.join(
        SYNTHESIZED_NEGATIVE_SAMPLING_EXTRACTED_DATA_PATH,
        args.dataset,
        f"{args.split}.jsonl",
    )
    return args


if __name__ == "__main__":
    options = parse_args()
    main(options)
